use sensor_fusion::{MadgwickFilterf32, SensorFusion};
use vqm::Vector3f32;

fn main() {
    let mut madgwick = MadgwickFilterf32::new();
    let dt = 0.0001; // 1 millisecond

    // Mock sensor values, normally read from IMU
    let gyro_dps = Vector3f32::new(100.0, 20.0, 10.0);
    let acc = Vector3f32::new(0.1, 0.2, 0.9);

    // Fuse acc and gyro values, after converting gyro to radians/s
    let orientation = madgwick.fuse_acc_gyro(acc, gyro_dps.to_radians(), dt);

    let roll = orientation.calculate_roll_degrees();
    let pitch = orientation.calculate_pitch_degrees();
    let yaw = orientation.calculate_yaw_degrees();

    // Print out the Euler angles.
    println!("pitch={}, roll={}, yaw={}", pitch, roll, yaw);
}
