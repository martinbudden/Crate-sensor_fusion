#![warn(clippy::pedantic)]
#![warn(unused_results)]

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::{RngExt, rng};
use std::hint::black_box;

use sensor_fusion::{ComplementaryFilterf32, FuseAccGyro, FuseAccGyroMag, MadgwickFilterf32, MahonyFilterf32};
use vqm::Vector3f32;

fn bench_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter");

    let mut madgwick_filter = MadgwickFilterf32::new();
    let mut mahony_filter = MahonyFilterf32::new();
    let mut complementary_filter = ComplementaryFilterf32::new();
    let delta_t = 0.000_1_f32;

    _ = group.throughput(Throughput::Elements(1));

    _ = group.bench_function("madgwick", |b| {
        b.iter_batched(
            || {
                let a: [f32; 3] = rng().random();
                let g: [f32; 3] = rng().random();
                let acc = Vector3f32::from(a);
                let gyro_rps = Vector3f32::from(g);
                (acc, gyro_rps)
            },
            |(acc, gyro_rps)| {
                black_box((acc, gyro_rps)).fuse_acc_gyro_using(black_box(&mut madgwick_filter), black_box(delta_t))
            },
            BatchSize::SmallInput,
        );
    });

    _ = group.bench_function("madgwick_mag", |b| {
        b.iter_batched(
            || {
                let a: [f32; 3] = rng().random();
                let g: [f32; 3] = rng().random();
                let m: [f32; 3] = rng().random();
                let acc = Vector3f32::from(a);
                let gyro_rps = Vector3f32::from(g);
                let mag = Vector3f32::from(m);
                (acc, gyro_rps, mag)
            },
            |(acc, gyro_rps, mag)| {
                black_box((acc, gyro_rps, mag))
                    .fuse_acc_gyro_mag_using(black_box(&mut madgwick_filter), black_box(delta_t))
            },
            BatchSize::SmallInput,
        );
    });

    _ = group.bench_function("mahony", |b| {
        b.iter_batched(
            || {
                let a: [f32; 3] = rng().random();
                let g: [f32; 3] = rng().random();
                let acc = Vector3f32::from(a);
                let gyro_rps = Vector3f32::from(g);
                (acc, gyro_rps)
            },
            |(acc, gyro_rps)| {
                black_box((acc, gyro_rps)).fuse_acc_gyro_using(black_box(&mut mahony_filter), black_box(delta_t))
            },
            BatchSize::SmallInput,
        );
    });

    _ = group.bench_function("complementary", |b| {
        b.iter_batched(
            || {
                let a: [f32; 3] = rng().random();
                let g: [f32; 3] = rng().random();
                let acc = Vector3f32::from(a);
                let gyro_rps = Vector3f32::from(g);
                (acc, gyro_rps)
            },
            |(acc, gyro_rps)| {
                black_box((acc, gyro_rps)).fuse_acc_gyro_using(black_box(&mut complementary_filter), black_box(delta_t))
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_filter);
criterion_main!(benches);
