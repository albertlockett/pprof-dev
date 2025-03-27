use std::fs::File;
use std::io::Write;
use std::iter::repeat_with;
use std::sync::Arc;

use arrow::error::Result;
use arrow_array::{Float32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use lance::{dataset::WriteMode, index::vector::VectorIndexParams, Dataset};
use object_store::ObjectStore;
use lance::dataset::{ReadParams, WriteParams};
use lance_index::traits::DatasetIndexExt;
use lance_linalg::distance::MetricType;
use lance::io::{ObjectStoreParams, WrappingObjectStore};
use lance_arrow::FixedSizeListArrayExt;
use parking_lot::RwLock;
use pprof::{protos::Message, Profiler, ReportBuilder, ReportTiming, SampleType, SampleTypes};
use pprof_object_store::ProfilingObjectStore;
use rand::Rng;

#[tokio::main]
async fn main() {
    env_logger::init();
    let vector_dims = 1536;
    let rows = 20_000;
    let schema = Arc::new(create_schema(vector_dims));

    let record_batch = generate_data(rows, vector_dims, schema.clone()).unwrap();

    let reader = RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema);

    let mut write_params = WriteParams::default();
    write_params.mode = WriteMode::Overwrite;
    if write_params.store_params.is_none() {
        write_params.store_params = Some(ObjectStoreParams::default());
    }
    let store_params = write_params.store_params.as_mut().unwrap();

    let profile_os_wrapper = Arc::new(ProfilingObjectStoreWrapper::new());
    store_params.object_store_wrapper = Some(profile_os_wrapper.clone());

    let report_timing = ReportTiming::default();

    let mut ds = Dataset::write(reader, "~/Desktop/lance_datasets/test_pprof.lance", Some(write_params)).await.unwrap();

    let params = VectorIndexParams::ivf_pq(4, 8, 2, MetricType::L2, 1);
    ds.create_index(
        &["vector"],
        lance_index::IndexType::Vector,
        None,
        &params,
        true
    ).await.unwrap();

    let report_builder = ReportBuilder::new(
        &profile_os_wrapper.get_profile,
        report_timing.clone(),
        SampleTypes::new(vec![
            SampleType::new(
                "object_store_get".to_string(),
                pprof::Unit::Count,
            )
        ])
    );
    let report = report_builder.build().unwrap();
    let mut file = File::create("get_profile.pb").unwrap();
    let profile = report.pprof().unwrap();

    let mut content = Vec::new();
    profile.write_to_vec(&mut content).unwrap();
    file.write_all(&content).unwrap();

    let report_builder = ReportBuilder::new(
        &profile_os_wrapper.put_profile,
        report_timing,
        SampleTypes::new(vec![
            SampleType::new(
                "object_store_put".to_string(),
                pprof::Unit::Count,
            )
        ])
    );
    let report = report_builder.build().unwrap();
    let mut file = File::create("put_profile.pb").unwrap();
    let profile = report.pprof().unwrap();

    let mut content = Vec::new();
    profile.write_to_vec(&mut content).unwrap();
    file.write_all(&content).unwrap();

}


fn create_schema(vector_dims: i32) -> Schema {
    let fields = vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            vector_dims,
        ),
        false,
    )];

    Schema::new(fields)
}

fn generate_data(rows: i32, vector_dims: i32, schema: Arc<Schema>) -> Result<RecordBatch> {
    let mut rng = rand::thread_rng();
    let vector_data = Float32Array::from_iter_values(
        repeat_with(|| rng.gen::<f32>()).take((vector_dims * rows) as usize),
    );

    let vectors = Arc::new(
        <arrow_array::FixedSizeListArray as FixedSizeListArrayExt>::try_new_from_values(
            vector_data,
            vector_dims,
        )
        .unwrap(),
    );

    Ok(RecordBatch::try_new(schema, vec![vectors])?)
}

struct ProfilingObjectStoreWrapper {
    get_profile: Arc<RwLock<pprof::Result<Profiler>>>,
    put_profile: Arc<RwLock<pprof::Result<Profiler>>>,
}

impl std::fmt::Debug for ProfilingObjectStoreWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ProfilingObjectStoreWrapper{}")?; // TODO?
        Ok(())
    }
}

impl ProfilingObjectStoreWrapper {
    fn new() -> Self {
        Self {
            // TODO is this a dumb way to initialize this? at least rethink it
            // TODO no unwrap
            get_profile: Arc::new(RwLock::new(Ok(Profiler::new().unwrap()))),
            put_profile: Arc::new(RwLock::new(Ok(Profiler::new().unwrap()))),
        }
    }
}

impl WrappingObjectStore for ProfilingObjectStoreWrapper {
    fn wrap(&self, original: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
        println!("wrapping the object store");
        Arc::new(ProfilingObjectStore {
            inner: original,
            get_profiler: self.get_profile.clone(),
            put_profiler: self.put_profile.clone(),
        })
    }
}