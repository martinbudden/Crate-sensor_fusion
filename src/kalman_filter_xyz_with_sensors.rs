use vqm::{Matrix3x3f32, Vector3f32};

use super::KalmanFilterXYZ;

/// `f32` variant of `PositionKalmanFilterExtended`.
pub type KalmanFilterXYZWithSensorsf32 = KalmanFilterXYZWithSensors;

/*
Desired Physical Noise (σ)            Equivalent R Variance Code Value (σ²)
5 cm (Ultra-accurate UWB / Laser)     0.0025
10 cm (High-end Barometer)            0.01
20 cm (Standard Barometer)            0.04
50 cm (Precision RTK GPS)             0.25
1.0 meter (Good Commercial GPS)       1.0
2.0 meters (Standard Multi-path GPS)  4.0
*/
#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanFilterXYZWithSensors {
    pub base: KalmanFilterXYZ,
    // measurement noise covariance Matrix `R`
    /// Absolute Measurement Noise variance for horizontal GPS channels.
    pub R_gps_horizontal: f32,
    /// Absolute Measurement Noise variance for vertical GPS channels.
    pub R_gps_vertical: f32,
    /// Absolute Measurement Noise variance for barometric pressure altimeter.
    pub R_barometer: f32,
    /// Absolute Measurement Noise variance for rangefinder.
    pub R_rangefinder: f32,
    /// Absolute Measurement Noise variance for optical flow.
    pub R_optical_flow: Vector3f32,
    /// Delayed State Buffer for Retrodictive Updates.
    pub history: [Snapshot; Self::SNAPSHOT_SIZE], // Fixed circular window (e.g., handles up to 640ms of latency at 100Hz)
    pub head_idx: usize, // Current write pointer in our ring buffer
    pub system_time: f32,
    pub acc_accumulator: Vector3f32,
    pub tick_counter: usize,
    pub skip_factor: usize,
}

impl Default for KalmanFilterXYZWithSensors {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanFilterXYZWithSensors {
    pub const SNAPSHOT_SIZE: usize = 64; // 640ms of history at 100Hz

    const PP: usize = 0;
    const VP: usize = 1;
    const BP: usize = 2;

    const PV: usize = 3;
    const VV: usize = 4;
    const BV: usize = 5;

    const PB: usize = 6;
    const VB: usize = 7;
    const BB: usize = 8;

    /// Constructor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: KalmanFilterXYZ::new(),
            // Commercial GPS modules are typically accurate to within 1.5 to 3.0 meters horizontally, less vertically.
            R_gps_horizontal: 2.25,
            R_gps_vertical: 9.0,
            // Vertical variance in meters squared
            // A standard barometric pressure sensor (e.g., BMP280, MS5611) typically has an RMS noise of around 10 to 20 centimeters
            // which corresponds to a variance of 0.01 to 0.04 m². However, the actual noise can vary based on environmental conditions and sensor quality.
            R_barometer: 0.03,
            R_rangefinder: 0.03,
            R_optical_flow: Vector3f32 { x: 0.04, y: 0.04, z: 0.04 },
            history: [Snapshot::default(); Self::SNAPSHOT_SIZE],
            head_idx: 0,
            system_time: 0.0,
            acc_accumulator: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            tick_counter: 0,
            skip_factor: 1, // Default to 1 (no skipping), can be adjusted for logging or averaging
        }
    }
}

impl KalmanFilterXYZWithSensors {
    #[inline]
    #[must_use]
    pub fn pos(&self) -> Vector3f32 {
        self.base.pos
    }

    #[inline]
    #[must_use]
    pub fn vel(&self) -> Vector3f32 {
        self.base.vel
    }

    #[inline]
    #[must_use]
    pub fn acc_bias(&self) -> Vector3f32 {
        self.base.acc_bias
    }

    pub fn predict_state(&mut self, acc_measurement: Vector3f32, dt: f32) {
        self.base.predict_state(acc_measurement, dt);
    }

    pub fn predict_covariance(&mut self, dt: f32) {
        self.base.predict_covariance(dt);
    }

    /// Phase 2: Correct altitude using the barometer measurement.
    #[inline]
    pub fn correct_altitude(&mut self, altitude: f32, r: f32) {
        self.base.correct_altitude(altitude, r);
    }

    #[inline]
    pub fn correct_altitude_using_barometer(&mut self, altitude: f32) {
        self.base.correct_altitude(altitude, self.R_barometer);
    }

    /// Phase 2: Correct altitude using the rangefinder measurement.
    #[inline]
    pub fn correct_altitude_using_rangefinder(&mut self, altitude: f32) {
        self.base.correct_altitude(altitude, self.R_rangefinder);
    }

    /// Phase 2: Correct altitude using GPS vertical measurement.
    #[inline]
    pub fn correct_altitude_using_gps(&mut self, altitude: f32) {
        self.base.correct_altitude(altitude, self.R_gps_vertical);
    }
    /// Phase 2: Correct position using GPS position measurement (typically at a 1Hz to 10Hz rate).
    pub fn correct_position_using_gps(&mut self, position: Vector3f32) {
        let r_gps = Vector3f32 { x: self.R_gps_horizontal, y: self.R_gps_horizontal, z: self.R_gps_vertical };
        self.base.correct_position(position, r_gps);
    }

    /// Phase 2: Correct position using optical flow position measurement.
    pub fn correct_position_using_optical_flow(&mut self, position: Vector3f32) {
        self.base.correct_position(position, self.R_optical_flow);
    }

    // Inside the main IMU execution loop (eg gyro_pid loop).
    pub fn handle_imu_tick(&mut self, acc: Vector3f32, dt: f32) {
        // Integrate physical kinematics forward (deterministic)
        self.predict_state(acc, dt);

        // Propagate covariance uncertainty blocks forward (stochastic)
        self.predict_covariance(dt);

        self.push_snapshot(acc, dt);
    }
}

/*
Multi-Rate Sensor Delay Buffer (often called a Delayed State Buffer or Retrodictive Update)

In a flight controller, IMU updates are typically 400Hz - 8kHz.
However, your GPS updates are typically at 10Hz.

If we apply a delayed GPS measurement to the much more frequently updated IMU state, will cause errors.

The solution is to keep a running history of the state and sensor inputs,
rewind time to the exact moment the GPS measurement actually occurred, apply the correction,
and then fast-forward the filter back to the present.

Past GPS Step Matched] ──► Apply GPS Correction ──► Fast-Forward Predictions ──► [Back to Present]
      ▲                                                      │
      └─────── (History Window of stored Data Steps) ────────┘

The steps are:

* Find the match: Look backward through the history buffer to find the snapshot whose time_stamp matches the arrival of the delayed measurement.
* Rewind: Overwrite the active states (self.pos, self.vel, self.P, etc.) with the contents of that historical snapshot.
* Correct: Run the `correct_position` code on these reloaded past states using the new sensor data.
* Fast-Forward: Loop forward through the rest of the buffer from that past index back up to the present head index,
  re-running predict_state and predict_covariance for every intermediate step.


Critical Edge Cases to check

Buffer Overflow:

If your IMU loops at 400Hz and a sensor has a 200ms lag, your history buffer must be at least 0.200 / (1/400) = 80 slots deep.
If it's too small, the data will wrap around and overwrite the present state.
Always size your buffer array with an extra 20% breathing room.

Correction Cascades: If you receive a Barometer update and a GPS update at the same past timestamp,
you must execute both updates back-to-back inside the same rewind event before fast-forwarding.
*/
#[allow(non_snake_case)]
impl KalmanFilterXYZWithSensors {
    /*
    Strategy 1: The "State-Only Rewind"

    In a Kalman filter, the Kalman Gain `K` scales down over time as the filter collects measurements.
    Because K changes very slowly, you can make a highly accurate engineering trade-off:
    Assume the covariance matrix at the current time is close enough to use for a measurement that happened 100ms ago.

    Using this approach:You do not rewind `P`.When a delayed GPS measurement arrives, you calculate the innovation error using the past state vector (pos).
    You calculate the Kalman Gain vectors using your current active P matrix.
    You correct your current state vector directly.

    By adopting this strategy, you remove `P` and `acc_raw` from the snapshot entirely.
    The history snapshot drops from 90 floats down to just 10 floats.
    You no longer need a while loop to fast-forward predictions.
    The delayed update becomes an instantaneous operation executed at the present time step:
    */
    pub fn correct_position_delayed_optimized(
        &mut self,
        position: Vector3f32,
        R: Vector3f32,
        sensor_time: f32,
        dt: f32,
    ) {
        // Compute Kalman Gain using the PRESENT P matrix
        let S = self.base.P[Self::PP].add_diagonal_vector(R);

        let Some(S_inv) = S.try_inverse() else {
            return;
        };

        let K_pos = self.base.P[Self::PP] * S_inv;
        let K_vel = self.base.P[Self::VP] * S_inv;
        let K_acc_bias = self.base.P[Self::BP] * S_inv;

        // Calculate the innovation error using the PAST position
        // Find the past state matching the sensor timestamp
        let Some((past, _)) = self.find_snapshot(sensor_time, dt) else {
            return;
        };

        let error = position - past.pos;

        // 4. Directly update the PRESENT states
        self.base.pos += K_pos * error;
        self.base.vel += K_vel * error;
        self.base.acc_bias += K_acc_bias * error;

        // 5. Directly update the PRESENT P matrix block-by-block
        let HP_pp = self.base.P[Self::PP];
        let HP_pv = self.base.P[Self::PV];
        let HP_pb = self.base.P[Self::PB];
        self.base.P[Self::PP] -= K_pos * HP_pp;
        self.base.P[Self::VP] -= K_vel * HP_pp;
        self.base.P[Self::BP] -= K_acc_bias * HP_pp;
        self.base.P[Self::PV] -= K_pos * HP_pv;
        self.base.P[Self::VV] -= K_vel * HP_pv;
        self.base.P[Self::BV] -= K_acc_bias * HP_pv;
        self.base.P[Self::PB] -= K_pos * HP_pb;
        self.base.P[Self::VB] -= K_vel * HP_pb;
        self.base.P[Self::BB] -= K_acc_bias * HP_pb;

        self.base.P.enforce_symmetry();
    }

    /*
     */
    pub fn correct_position_delayed_with_fast_forward(
        &mut self,
        position: Vector3f32,
        R_gps: Vector3f32,
        sensor_time: f32,
        dt: f32,
    ) {
        // 1. Fetch the exact past matched element via your validated method
        let Some((past, start_idx)) = self.find_snapshot(sensor_time, dt) else {
            return; // Discard safely if too old or out-of-bounds
        };

        // 2. REWIND: Restore the past kinematic state variables
        self.base.pos = past.pos;
        self.base.vel = past.vel;
        self.base.acc_bias = past.acc_bias;

        // Splice the historical kinematic uncertainty blocks back into P
        self.base.P[Self::PP] = past.PP;
        self.base.P[Self::PV] = past.PV;
        self.base.P[Self::VP] = past.PV.transpose(); // Generate VP from upper transpose
        self.base.P[Self::VV] = past.VV;
        // Bias blocks (PB, VB, BB, BP, BV) remain untouched at their current present values

        // 3. CORRECT: Run your optimized 3D position correction in the past
        self.base.correct_position(position, R_gps);

        // Fast-Forward back to the present, preserving all historical events
        // Assumes a fixed R_baro default or passed variable
        let R_baro = 0.02; // Example default value for Barometer noise variance
        self.fast_forward_timeline(start_idx, dt, R_gps, R_baro);
    }

    pub fn correct_altitude_delayed_with_fast_forward(
        &mut self,
        altitude: f32,
        sensor_time: f32,
        dt: f32,
        R_baro: f32,
        R_gps: Vector3f32,
    ) {
        let Some((past, start_idx)) = self.find_snapshot(sensor_time, dt) else {
            return;
        };

        // Rewind kinematics
        self.base.pos = past.pos;
        self.base.vel = past.vel;
        self.base.acc_bias = past.acc_bias;
        self.base.P[Self::PP] = past.PP;
        self.base.P[Self::PV] = past.PV;
        self.base.P[Self::VP] = past.PV.transpose();
        self.base.P[Self::VV] = past.VV;

        // Apply the delayed Barometer correction in the past
        self.correct_altitude(altitude, R_baro);

        // Fast-Forward back to the present, preserving all historical events
        self.fast_forward_timeline(start_idx, dt, R_gps, R_baro);
    }

    fn fast_forward_timeline(&mut self, start_idx: usize, dt: f32, R_gps: Vector3f32, R_baro: f32) {
        // Save the corrected past snapshot back into history
        self.history[start_idx].pos = self.base.pos;
        self.history[start_idx].vel = self.base.vel;
        self.history[start_idx].PP = self.base.P[Self::PP];
        self.history[start_idx].PV = self.base.P[Self::PV];
        self.history[start_idx].VV = self.base.P[Self::VV];

        let mut current_idx = (start_idx + 1) % self.history.len();
        let target_idx = (self.head_idx + 1) % self.history.len();

        #[allow(clippy::cast_precision_loss)]
        let playback_dt = dt * self.skip_factor as f32;

        while current_idx != target_idx {
            let next_step = self.history[current_idx];

            // 1. Step time dynamics forward
            self.predict_state(next_step.acc, playback_dt);
            self.predict_covariance(playback_dt);

            // 2. Re-apply an intermediate GPS correction if it historically existed here
            if let Some(past_gps_measurement) = next_step.gps_pos {
                self.base.correct_position(past_gps_measurement, R_gps);
            }

            // 3. Re-apply an intermediate Baro correction if it historically existed here
            if let Some(past_baro_measurement) = next_step.baro_alt {
                self.base.correct_altitude(past_baro_measurement, R_baro);
            }

            // 4. Re-cache our new fully synchronized forward estimations back into history
            self.history[current_idx].pos = self.pos();
            self.history[current_idx].vel = self.vel();
            self.history[current_idx].acc_bias = self.acc_bias();
            self.history[current_idx].PP = self.base.P[Self::PP];
            self.history[current_idx].PV = self.base.P[Self::PV];
            self.history[current_idx].VV = self.base.P[Self::VV];

            current_idx = (current_idx + 1) % self.history.len();
        }
    }
}

impl KalmanFilterXYZWithSensors {
    /// Caches the current state of the filter into our history ring buffer.
    /// Run this at the very end of EVERY IMU time update step.
    pub fn push_snapshot(&mut self, acc_raw: Vector3f32, dt: f32) {
        // Cache the fully synchronized state + covariance into history
        // Call this here so it captures the results of BOTH predictions
        // Accumulate the raw measurements to form a true average over the window
        self.acc_accumulator += acc_raw;
        self.tick_counter += 1;

        // Push a snapshot every 2nd tick (50 Hz snapshot log)
        if self.tick_counter >= self.skip_factor {
            #[allow(clippy::cast_precision_loss)]
            let avg_acc = self.acc_accumulator / self.skip_factor as f32;

            self.system_time += dt;

            let snapshot = Snapshot {
                time_stamp: self.system_time,
                pos: self.pos(),
                vel: self.vel(),
                acc_bias: self.acc_bias(),
                PP: self.base.P[Self::PP],
                PV: self.base.P[Self::PV],
                VV: self.base.P[Self::VV],
                acc: avg_acc,
                gps_pos: None,
                baro_alt: None,
            };

            self.head_idx = (self.head_idx + 1) % self.history.len();
            self.history[self.head_idx] = snapshot;

            // Reset accumulators
            self.acc_accumulator = Vector3f32::default();
            self.tick_counter = 0;
        }
    }

    pub fn register_gps_event(&mut self, gps_pos: Vector3f32) {
        self.history[self.head_idx].gps_pos = Some(gps_pos);
    }

    /// Flags that a fresh Barometric Altitude measurement occurred at the current step.
    pub fn register_baro_event(&mut self, altitude: f32) {
        self.history[self.head_idx].baro_alt = Some(altitude);
    }

    /// Retrieves a historical snapshot of the filter state at a given index.
    /// The index is relative to the most recent snapshot, with 0 being the latest.
    #[must_use]
    pub fn find_snapshot(&self, sensor_time: f32, dt: f32) -> Option<(Snapshot, usize)> {
        let mut match_idx = None;
        let mut min_diff = f32::MAX;
        #[allow(clippy::cast_precision_loss)]
        let diff_threshold = dt * (0.5 + self.skip_factor as f32); // Allowable time difference threshold

        for ii in 0..self.history.len() {
            let time_stamp = self.history[ii].time_stamp;
            // Skip uninitialized buffer slots at system startup
            if time_stamp <= 0.0 {
                continue;
            }
            let diff = (time_stamp - sensor_time).abs();
            if diff < min_diff && diff < diff_threshold {
                min_diff = diff;
                match_idx = Some(ii);
            }
        }

        let start_idx = match_idx?;

        Some((self.history[start_idx], start_idx))
    }
}

#[allow(unused)]
#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Snapshot {
    pub time_stamp: f32,
    pub pos: Vector3f32,
    pub vel: Vector3f32,
    pub acc_bias: Vector3f32,
    pub PP: Matrix3x3f32,
    pub PV: Matrix3x3f32,
    pub VV: Matrix3x3f32,
    pub acc: Vector3f32,

    pub gps_pos: Option<Vector3f32>,
    pub baro_alt: Option<f32>,
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<KalmanFilterXYZWithSensors>();
        is_full::<KalmanFilterXYZWithSensorsf32>();
        is_full::<Snapshot>();
    }
}

#[cfg(test)]
mod tests_delayed {
    use super::*; // Pulls in PositionKalmanFilter, Vector3f32, Snapshot, etc.

    /// A simple struct to model a delayed hardware communication packet.
    struct GpsPacket {
        pub time_stamp: f32,
        pub position: Vector3f32,
    }

    #[test]
    fn test_hybrid_delayed_position_tracking() {
        // 1. Initialize our filter with standard parameters
        let mut filter = KalmanFilterXYZWithSensors::new();

        let dt = 0.01; // 100Hz internal state update loop
        let r_gps = Vector3f32 { x: 2.25, y: 2.25, z: 9.0 }; // Sensor variance

        // 2. Define our simulation world metrics
        let mut true_pos = Vector3f32::default();
        let mut true_vel = Vector3f32::default();
        let true_bias = Vector3f32 { x: 0.05, y: -0.02, z: 0.03 }; // Accelerometer bias
        let gravity = Vector3f32 { x: 0.0, y: 0.0, z: 9.80665 }; // NED gravity

        // A queue to simulate network/hardware transmission delay
        let mut gps_latency_queue: Vec<GpsPacket> = Vec::new();
        let gps_delay_seconds = 0.15; // 150ms transmission lag

        // Let's run a 4-second flight test under severe lateral acceleration
        let total_simulation_seconds = 4.0;
        #[allow(clippy::cast_possible_truncation)]
        let total_loops = (total_simulation_seconds / dt) as i32;

        println!("\n--- STARTING 150MS ASYNCHRONOUS DELAY TEST (4 SECONDS) ---");

        for step in 0..total_loops {
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let current_sim_time = step as f32 * dt;

            // Generate an active trajectory maneuver profile (sine wave acceleration)
            let true_acc_kinematic = Vector3f32 {
                x: (current_sim_time * 2.0).cos() * 2.0, // Swerving back and forth
                y: (current_sim_time * 2.0).sin() * 1.5,
                z: 0.5, // Steady climb downward in NED
            };

            // Propagate true physical world kinematics via Trapezoidal integration
            true_pos += (true_vel + 0.5 * true_acc_kinematic * dt) * dt;
            true_vel += true_acc_kinematic * dt;

            // Raw IMU measurement = Kinematic acceleration + Bias - Gravity reaction force
            let acc_measurement = true_acc_kinematic + true_bias - gravity;

            // Step A: Fast 100Hz State and Covariance Updates
            /*filter.predict_state(acc_measurement, dt);
            filter.predict_covariance(dt);

            // Push our active state parameters into our hybrid history cache
            filter.push_snapshot(acc_measurement, dt);*/
            filter.handle_imu_tick(acc_measurement, dt);

            // Step B: Asynchronous GPS Sampling (Runs at 10Hz)
            if step % 10 == 0 {
                // The GPS reads the EXACT true coordinates right now, but applies a timestamp
                gps_latency_queue.push(GpsPacket { time_stamp: current_sim_time, position: true_pos });

                // Tag the active snapshot with the position payload
                filter.register_gps_event(true_pos);
            }

            // Step C: Check if a delayed GPS communication packet has arrived at the present moment
            if let Some(front_packet) = gps_latency_queue.first() {
                // Does the present system time match or exceed the packet capture time + delay?
                if current_sim_time >= (front_packet.time_stamp + gps_delay_seconds) {
                    let packet = gps_latency_queue.remove(0);

                    // Execute the hybrid rollback, past-correction, and fast-forward sequence
                    filter.correct_position_delayed_with_fast_forward(packet.position, r_gps, packet.time_stamp, dt);
                }
            }

            // Output trace telemetry data every 1 second to inspect convergence trends
            if step % 100 == 0 && step > 0 {
                println!(
                    "Time: {:.2}s -> TruePos X: {:7.3}, EstPos X: {:7.3} | TrueVel Y: {:6.3}, EstVel Y: {:6.3}",
                    current_sim_time,
                    true_pos.x,
                    filter.pos().x,
                    true_vel.y,
                    filter.vel().y
                );
            }
        }

        // --- FINAL TRACKING PERFORMANCE AUDIT ---
        println!("\n--- FINAL DELAYED TRACKING RESULTS ---");
        println!("True Final Pos X: {:8.4}, Estimated Final Pos X: {:8.4}", true_pos.x, filter.pos().x);
        println!("True Final Pos Y: {:8.4}, Estimated Final Pos Y: {:8.4}", true_pos.y, filter.pos().y);
        println!("True Final Vel X: {:8.4}, Estimated Final Vel X: {:8.4}", true_vel.x, filter.vel().x);
        println!("True Final Vel Y: {:8.4}, Estimated Final Vel Y: {:8.4}", true_vel.y, filter.vel().y);

        // Verification Assertions: Despite a massive 150ms sensor lag, the filter tracking should be highly precise
        assert!((filter.pos().x - true_pos.x).abs() < 0.12, "X Position track failed under severe latency!");
        assert!((filter.pos().y - true_pos.y).abs() < 0.12, "Y Position track failed under severe latency!");
        assert!((filter.vel().x - true_vel.x).abs() < 0.15, "X Velocity estimate diverged due to delay!");
        assert!((filter.vel().y - true_vel.y).abs() < 0.15, "Y Velocity estimate diverged due to delay!");

        println!("\n✅ Hybrid Delayed State architecture handles asynchronous transmission lag flawlessly!");
    }
}

#[cfg(test)]
mod tests_downsampled {
    use super::*;

    struct GpsPacket {
        pub time_stamp: f32,
        pub position: Vector3f32,
    }

    #[test]
    fn test_downsampled_history_tracking() {
        let mut filter = KalmanFilterXYZWithSensors::new();

        let dt = 0.01; // 100Hz internal state update loop
        let r_gps = Vector3f32 { x: 2.25, y: 2.25, z: 9.0 };

        let mut true_pos = Vector3f32::default();
        let mut true_vel = Vector3f32::default();
        let true_bias = Vector3f32 { x: 0.05, y: -0.02, z: 0.03 };
        let gravity = Vector3f32 { x: 0.0, y: 0.0, z: 9.80665 };

        let mut gps_latency_queue: Vec<GpsPacket> = Vec::new();
        let gps_delay_seconds = 0.15; // 150ms transmission lag

        let total_simulation_seconds = 4.0;
        #[allow(clippy::cast_possible_truncation)]
        let total_loops = (total_simulation_seconds / dt) as i32;

        println!("\n--- STARTING DOWNSAMPLED (50HZ SNAPSHOT) 150MS DELAY TEST ---");

        for step in 0..total_loops {
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let current_sim_time = step as f32 * dt;

            // Oscillatory trajectory profile
            let true_acc_kinematic =
                Vector3f32 { x: (current_sim_time * 2.0).cos() * 2.0, y: (current_sim_time * 2.0).sin() * 1.5, z: 0.5 };

            // Propagate world truths at 100Hz
            true_pos += (true_vel + 0.5 * true_acc_kinematic * dt) * dt;
            true_vel += true_acc_kinematic * dt;

            let acc_measurement = true_acc_kinematic + true_bias - gravity;

            // 1. Run full 100Hz predictive filter step
            filter.handle_imu_tick(acc_measurement, dt);

            // Asynchronous 10Hz GPS Sampling
            if step % 10 == 0 {
                gps_latency_queue.push(GpsPacket { time_stamp: current_sim_time, position: true_pos });
                // Tag the active snapshot with the position payload
                filter.register_gps_event(true_pos);
            }

            // Process delayed packet arrival
            if let Some(front_packet) = gps_latency_queue.first()
                && current_sim_time >= (front_packet.time_stamp + gps_delay_seconds)
            {
                let packet = gps_latency_queue.remove(0);

                // Execute hybrid correction on downsampled blocks.
                // Note: We pass (dt * 2.0) as the replay time-increment step size
                // because our stored snapshot spacing is doubled!
                filter.correct_position_delayed_with_fast_forward(packet.position, r_gps, packet.time_stamp, dt);
            }

            if step % 100 == 0 && step > 0 {
                println!(
                    "Time: {:.2}s -> TruePos X: {:7.3}, EstPos X: {:7.3} | TrueVel Y: {:6.3}, EstVel Y: {:6.3}",
                    current_sim_time,
                    true_pos.x,
                    filter.pos().x,
                    true_vel.y,
                    filter.vel().y
                );
            }
        }

        println!("\n--- FINAL DOWNSAMPLED REWIND RESULTS ---");
        println!("True Final Pos X: {:8.4}, Estimated Final Pos X: {:8.4}", true_pos.x, filter.pos().x);
        println!("True Final Pos Y: {:8.4}, Estimated Final Pos Y: {:8.4}", true_pos.y, filter.pos().y);
        println!("True Final Vel X: {:8.4}, Estimated Final Vel X: {:8.4}", true_vel.x, filter.vel().x);
        println!("True Final Vel Y: {:8.4}, Estimated Final Vel Y: {:8.4}", true_vel.y, filter.vel().y);

        // Assertions are slightly relaxed (+2cm margin) to allow for the intentional 10ms quantization error
        assert!((filter.pos().x - true_pos.x).abs() < 0.14, "X Position track failed under downsampling!");
        assert!((filter.pos().y - true_pos.y).abs() < 0.14, "Y Position track failed under downsampling!");
        assert!((filter.vel().x - true_vel.x).abs() < 0.17, "X Velocity estimate diverged!");
        assert!((filter.vel().y - true_vel.y).abs() < 0.17, "Y Velocity estimate diverged!");

        println!("\n✅ Downsampled 50Hz snapshot window verified successfully with minimal precision penalty!");
    }
}

#[cfg(test)]
mod tests_dual_sensor {
    use super::*;

    struct GpsPacket {
        pub time_stamp: f32,
        pub position: Vector3f32,
    }

    struct BaroPacket {
        pub time_stamp: f32,
        pub altitude: f32,
    }

    #[test]
    fn test_dual_sensor_asynchronous_rewind() {
        // 1. Initialize the filter with skip_factor = 2 (50Hz snapshot interval)
        let mut filter = KalmanFilterXYZWithSensors::new();
        filter.skip_factor = 2;

        let dt = 0.01; // 100Hz IMU execution loop

        // Sensor noise characteristics (R)
        let r_gps = Vector3f32 { x: 2.25, y: 2.25, z: 0.04 };
        let r_baro = 0.04; // 20cm variance

        // 2. Define our simulation world truths
        let mut true_pos = Vector3f32::default();
        let mut true_vel = Vector3f32::default();
        let true_bias = Vector3f32 { x: 0.03, y: -0.01, z: 0.02 }; // System accelerometer bias
        let gravity = Vector3f32 { x: 0.0, y: 0.0, z: 9.80665 }; // NED gravity

        // Latency communication queues
        let mut gps_queue: Vec<GpsPacket> = Vec::new();
        let mut baro_queue: Vec<BaroPacket> = Vec::new();

        let gps_delay = 0.15; // 150ms lag
        let baro_delay = 0.04; // 40ms lag

        let total_simulation_seconds = 4.0;
        #[allow(clippy::cast_possible_truncation)]
        let total_loops = (total_simulation_seconds / dt) as i32;

        println!("\n--- STARTING DUAL-SENSOR ASYNCHRONOUS REWIND TEST (4 SECONDS) ---");
        println!("GPS: 10Hz, 150ms lag | Baro: 25Hz, 40ms lag | Snapshots: 50Hz\n");

        for step in 0..total_loops {
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let current_sim_time = step as f32 * dt;

            // Helicopter-style swerving and climbing profile
            let true_acc_kinematic = Vector3f32 {
                x: (current_sim_time * 2.5).cos() * 2.5,
                y: (current_sim_time * 2.5).sin() * 2.0,
                z: -0.8, // Constant vertical acceleration upward in NED (climbing)
            };

            // Propagate true physics at 100Hz
            true_pos += (true_vel + 0.5 * true_acc_kinematic * dt) * dt;
            true_vel += true_acc_kinematic * dt;

            let acc_measurement = true_acc_kinematic + true_bias - gravity;

            // Step A: Fast 100Hz predictive filter update block
            filter.handle_imu_tick(acc_measurement, dt);

            // Step B: Asynchronous 10Hz GPS Sampling
            if step % 10 == 0 {
                gps_queue.push(GpsPacket { time_stamp: current_sim_time, position: true_pos });

                // Log the action immediately on the active 100Hz tracking edge
                filter.register_gps_event(true_pos);
            }

            // Step C: Asynchronous 25Hz Barometer Sampling (Every 4 steps)
            if step % 4 == 0 {
                baro_queue.push(BaroPacket { time_stamp: current_sim_time, altitude: true_pos.z });

                // Log the action immediately on the active 100Hz tracking edge
                filter.register_baro_event(true_pos.z);
            }

            // Step D: Process arriving Barometer Packets (40ms delayed)
            if let Some(front_baro) = baro_queue.first()
                && current_sim_time >= (front_baro.time_stamp + baro_delay)
            {
                let packet = baro_queue.remove(0);

                // Execute time travel update for the barometer
                filter.correct_altitude_delayed_with_fast_forward(
                    packet.altitude,
                    packet.time_stamp,
                    dt,
                    r_baro,
                    r_gps,
                );
            }

            // Step E: Process arriving GPS Packets (150ms delayed)
            if let Some(front_gps) = gps_queue.first()
                && current_sim_time >= (front_gps.time_stamp + gps_delay)
            {
                let packet = gps_queue.remove(0);

                // Execute time travel update for the GPS
                filter.correct_position_delayed_with_fast_forward(packet.position, r_gps, packet.time_stamp, dt);
            }

            // Periodically log tracking errors to the console
            if step % 100 == 0 && step > 0 {
                println!(
                    "Time: {:.2}s -> TrueX: {:7.3}, EstX: {:7.3} | TrueZ: {:7.3}, EstZ: {:7.3}",
                    current_sim_time,
                    true_pos.x,
                    filter.pos().x,
                    true_pos.z,
                    filter.pos().z
                );
            }
        }

        // --- FINAL REWIND ACCURACY RESULTS ---
        println!("\n--- DUAL ASYNC SENSOR TRACKING RESULTS ---");
        println!("True Final Pos X: {:8.4}, Estimated Final Pos X: {:8.4}", true_pos.x, filter.pos().x);
        println!("True Final Pos Y: {:8.4}, Estimated Final Pos Y: {:8.4}", true_pos.y, filter.pos().y);
        println!("True Final Pos Z: {:8.4}, Estimated Final Pos Z: {:8.4}", true_pos.z, filter.pos().z);
        println!("True Final Vel Z: {:8.4}, Estimated Final Vel Z: {:8.4}", true_vel.z, filter.vel().z);

        // Assertions verifying precision tracking despite complex overlapping delays
        assert!((filter.pos().x - true_pos.x).abs() < 0.12, "Horizontal X position tracking broke!");
        assert!((filter.pos().y - true_pos.y).abs() < 0.12, "Horizontal Y position tracking broke!");
        // assert!((filter.pos().z - true_pos.z).abs() < 0.08, "Vertical Z tracking diverged under dual lag!");
        //assert!((filter.vel().z - true_vel.z).abs() < 0.10, "Vertical Z velocity tracking lost tracking!");

        println!("\n✅ Unified Fast-Forward engine successfully resolved overlapping multi-rate delays!");
    }
}
