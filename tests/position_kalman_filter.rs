use sensor_fusion::PositionKalmanFilter;
use vqm::{Matrix9x9f32, Vector3f32};

#[cfg(test)]
mod position_filter_tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /*
    The Core Magic of Kalman Filtering:

    Note that we passed only an altitude reading (110.0) into the function.
    Yet, because the matrix infrastructure maintains cross-covariance terms,
    the math automatically calculated how that altitude discrepancy should
    adjust the estimated vertical velocity and sensor bias registers simultaneously
    without writing explicit scalar formulas.

    Orthogonality Isolation:
    This test proves that a pure vertical sensor step doesn't bleed data into
    or corrupt the horizontal dimensions (pos.x, vel.y), verifying that the multi-sensor isolation layers are mathematically sound
    */
    #[test]
    fn test_barometer_measurement_update() {
        // Initialize a baseline filter instance with realistic test values
        let mut filter = PositionKalmanFilter {
            pos: Vector3f32 { x: 10.0, y: 20.0, z: 100.0 }, // Initial Altitude is 100m
            vel: Vector3f32 { x: 1.0, y: 2.0, z: 5.0 },     // Initial Vertical Velocity is 5m/s
            acc_bias: Vector3f32 { x: 0.01, y: 0.02, z: 0.1 }, // Initial Vertical Bias is 0.1m/s²

            // Setting up a predictable diagonal P matrix
            P: Matrix9x9f32::new({
                let mut d = [0.0; 81];
                // Fill indices corresponding to x, y, z variances
                d[Matrix9x9f32::M11] = 2.0; // X position variance
                d[Matrix9x9f32::M22] = 2.0; // Y position variance
                d[Matrix9x9f32::M33] = 4.0; // Z position variance (P_zz)

                d[Matrix9x9f32::M44] = 1.0; // X velocity variance
                d[Matrix9x9f32::M55] = 1.0; // Y velocity variance
                d[Matrix9x9f32::M66] = 2.0; // Z velocity variance

                d[Matrix9x9f32::M77] = 0.1; // X bias variance
                d[Matrix9x9f32::M88] = 0.1; // Y bias variance
                d[Matrix9x9f32::M99] = 0.5; // Z bias variance

                // Inject a cross-covariance between Z-position (3) and Z-velocity (6)
                // and Z-position (3) and Z-bias (9) to prove the matrix update paths works!
                d[Matrix9x9f32::M36] = 1.0; // Column 3, Row 6 (cross term)
                d[Matrix9x9f32::M39] = -0.5; // Column 3, Row 9 (cross term)
                d
            }),
            E: Matrix9x9f32::default(),
            q_velocity: 0.1,
            q_bias: 0.01,
            r_gps_horizontal: 1.0,
            r_gps_vertical: 2.0,
            r_barometer: 4.0, // Measurement noise R = 4.0
            r_rangefinder: 0.0,
            r_optical_flow: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
        };

        // Introduce an altitude measurement with a 10-meter error step
        // Expected: Innovation Error = 110.0 - 100.0 = 10.0
        let barometer_reading = 110.0;

        // Run the update loop
        // S = P_zz + R = 4.0 + 4.0 = 8.0
        // K_pos_z = P_zz / S = 4.0 / 8.0 = 0.5
        // K_vel_z = P_vz / S = 1.0 / 8.0 = 0.125
        // K_bias_z = P_bz / S = -0.5 / 8.0 = -0.0625
        filter.correct_altitude_using_barometer(barometer_reading);

        // Verify State Corrections: New Value = Old Value + (K * Error)
        // Expected Z position = 100.0 + (0.5 * 10.0) = 105.0
        assert_abs_diff_eq!(filter.pos.z, 105.0, epsilon = 1e-5);

        // Expected Z velocity = 5.0 + (0.125 * 10.0) = 6.25
        assert_abs_diff_eq!(filter.vel.z, 6.25, epsilon = 1e-5);

        // Expected Z bias = 0.1 + (-0.0625 * 10.0) = -0.525
        assert_abs_diff_eq!(filter.acc_bias.z, -0.525, epsilon = 1e-5);

        // Verify that Horizontal States remain completely untouched by the vertical sensor
        assert_abs_diff_eq!(filter.pos.x, 10.0, epsilon = 1e-5);
        assert_abs_diff_eq!(filter.vel.y, 2.0, epsilon = 1e-5);

        // Verify Covariance Matrix Extraction Reduction (E = P - K * H * P)
        // New P_zz (index 3,3) = 4.0 - (0.5 * 4.0) = 2.0
        assert_abs_diff_eq!(filter.E[Matrix9x9f32::M33], 2.0, epsilon = 1e-5);
    }
}

#[cfg(test)]
mod gps_filter_tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_gps_3d_measurement_update() {
        // Initialize a baseline filter instance with clean, predictable values
        let mut filter = PositionKalmanFilter {
            pos: Vector3f32 { x: 10.0, y: 20.0, z: 30.0 },   // Initial Position (m)
            vel: Vector3f32 { x: 1.0, y: 2.0, z: 3.0 },      // Initial Velocity (m/s)
            acc_bias: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 }, // Initial Accelerometer Bias

            // Setup a predictable P matrix layout
            // For educational clarity, we start with a clean diagonal covariance
            P: Matrix9x9f32::new({
                let mut d = [0.0; 81];
                // Fill the diagonal variances for Position (Rows 1-3)
                d[Matrix9x9f32::M11] = 2.0; // X variance
                d[Matrix9x9f32::M22] = 2.0; // Y variance
                d[Matrix9x9f32::M33] = 3.0; // Z variance

                // Fill diagonal variances for Velocity (Rows 4-6)
                d[Matrix9x9f32::M44] = 1.0;
                d[Matrix9x9f32::M55] = 1.0;
                d[Matrix9x9f32::M66] = 1.0;

                // Fill diagonal variances for Bias (Rows 7-9)
                d[Matrix9x9f32::M77] = 0.1;
                d[Matrix9x9f32::M88] = 0.1;
                d[Matrix9x9f32::M99] = 0.1;

                // Inject cross-covariances to test the multidimensional gain path:
                // Here, we simulate that an error in X-position correlates with X-velocity,
                // and an error in Z-position correlates with Z-bias.
                d[Matrix9x9f32::M14] = 0.5; // Row 4, Col 1 (X-vel vs X-pos)
                d[Matrix9x9f32::M39] = -0.2; // Row 9, Col 3 (Z-bias vs Z-pos)
                d
            }),
            E: Matrix9x9f32::default(),
            q_velocity: 0.1,
            q_bias: 0.01,
            r_gps_horizontal: 2.0, // Horizontal GPS noise covariance
            r_gps_vertical: 3.0,   // Vertical GPS noise covariance
            r_barometer: 1.0,
            r_rangefinder: 0.0,
            r_optical_flow: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
        };

        // Introduce a 3D GPS packet with a specific translation step error
        // Innovation Error = GPS - Predicted = [14.0 - 10.0, 20.0 - 20.0, 36.0 - 30.0]
        //                  = [4.0, 0.0, 6.0]
        let gps_reading = Vector3f32 { x: 14.0, y: 20.0, z: 36.0 };

        // Run the multidimensional update logic
        // Hand-calculating the top-left block behavior for verification:
        // S = P_pos + R_gps
        // S_xx = 2.0 (P_xx) + 2.0 (R_horizontal) = 4.0  => S_inv_xx = 1.0 / 4.0 = 0.25
        // S_zz = 3.0 (P_zz) + 3.0 (R_vertical)  = 6.0  => S_inv_zz = 1.0 / 6.0 = 0.166666
        // K_pos_xx = P_xx * S_inv_xx = 2.0 * 0.25 = 0.5
        // K_pos_zz = P_zz * S_inv_zz = 3.0 * 0.166666 = 0.5
        filter.correct_position_using_gps(gps_reading);

        // Verify Position Updates: New = Old + (K * Error)
        // New X pos = 10.0 + (0.5 * 4.0) + (0.0 * 0.0) + (0.0 * 6.0) = 12.0
        assert_abs_diff_eq!(filter.pos.x, 12.0, epsilon = 1e-5);

        // New Y pos = 20.0 + (0.5 * 0.0) = 20.0 (Error was 0, should stay locked)
        assert_abs_diff_eq!(filter.pos.y, 20.0, epsilon = 1e-5);

        // New Z pos = 30.0 + (0.5 * 6.0) = 33.0
        assert_abs_diff_eq!(filter.pos.z, 33.0, epsilon = 1e-5);

        // Verify Cross-State Corrections (Velocity and Bias channels)
        // K_vel_x = P_vx_px * S_inv_xx = 0.5 * 0.25 = 0.125
        // New X velocity = 1.0 + (0.125 * 4.0) = 1.5
        assert_abs_diff_eq!(filter.vel.x, 1.5, epsilon = 1e-5);

        // K_bias_z = P_bz_pz * S_inv_zz = -0.2 * 0.166666 = -0.033333
        // New Z bias = 0.0 + (-0.033333 * 6.0) = -0.2
        assert_abs_diff_eq!(filter.acc_bias.z, -0.2, epsilon = 1e-5);

        // Verify Post-Correction Covariance Matrix Changes (E = P - K * H * P)
        // New P_xx = 2.0 - (K_pos_xx * P_xx) = 2.0 - (0.5 * 2.0) = 1.0
        assert_abs_diff_eq!(filter.P[Matrix9x9f32::M11], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_gps_3d_measurement_update_with_tuple_blocks() {
        // Initialize a baseline filter instance with predictable state variances
        let mut filter = PositionKalmanFilter {
            pos: Vector3f32 { x: 10.0, y: 20.0, z: 30.0 },   // Initial Position (m)
            vel: Vector3f32 { x: 1.0, y: 2.0, z: 3.0 },      // Initial Velocity (m/s)
            acc_bias: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 }, // Initial Accelerometer Bias

            // Setup a predictable, diagonal-dominated covariance baseline
            P: Matrix9x9f32::new({
                let mut d = [0.0; 81];
                // Fill the diagonal variances for Position (Rows 1-3)
                d[Matrix9x9f32::M11] = 2.0; // X variance
                d[Matrix9x9f32::M22] = 2.0; // Y variance
                d[Matrix9x9f32::M33] = 3.0; // Z variance

                // Fill diagonal variances for Velocity (Rows 4-6)
                d[Matrix9x9f32::M44] = 1.0;
                d[Matrix9x9f32::M55] = 1.0;
                d[Matrix9x9f32::M66] = 1.0;

                // Fill diagonal variances for Bias (Rows 7-9)
                d[Matrix9x9f32::M77] = 0.1;
                d[Matrix9x9f32::M88] = 0.1;
                d[Matrix9x9f32::M99] = 0.1;

                // Inject cross-covariances to test the multidimensional gain path:
                // Correlation 1: X-position vs X-velocity
                d[Matrix9x9f32::M14] = 0.5; // Row 4, Col 1
                // Correlation 2: Z-position vs Z-bias
                d[Matrix9x9f32::M39] = -0.2; // Row 9, Col 3
                d
            }),
            E: Matrix9x9f32::default(),
            q_velocity: 0.1,
            q_bias: 0.01,
            r_gps_horizontal: 2.0, // Horizontal GPS noise covariance (R)
            r_gps_vertical: 3.0,   // Vertical GPS noise covariance (R)
            r_barometer: 1.0,
            r_rangefinder: 0.0,
            r_optical_flow: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
        };

        // Introduce a 3D GPS packet with a specific translation step error
        // Innovation Error = GPS - Predicted = [14.0 - 10.0, 20.0 - 20.0, 36.0 - 30.0]
        //                  = [4.0, 0.0, 6.0]
        let gps_reading = Vector3f32 { x: 14.0, y: 20.0, z: 36.0 };

        // Run the refactored multidimensional update logic
        filter.correct_position_using_gps(gps_reading);

        // Verify Position Updates: New = Old + (K_pos * Error)
        // Expected X pos = 10.0 + 2.0
        assert_abs_diff_eq!(filter.pos.x, 12.0, epsilon = 1e-5);

        // Expected Y pos = 20.0 + 0.0 (Error was 0)
        assert_abs_diff_eq!(filter.pos.y, 20.0, epsilon = 1e-5);

        // Expected Z pos = 30.0 + 3.0
        assert_abs_diff_eq!(filter.pos.z, 33.0, epsilon = 1e-5);

        // Verify Cross-State Corrections via K_vel and K_acc_bias
        // New X velocity = 1.0 + 0.5
        assert_abs_diff_eq!(filter.vel.x, 1.5, epsilon = 1e-5);

        // New Z bias = 0.0 - 0.2
        assert_abs_diff_eq!(filter.acc_bias.z, -0.2, epsilon = 1e-5);

        // Verify Post-Correction Covariance Matrix Changes (E = P - KH_P)
        // New P_xx = 2.0 - 1.0 = 1.0
        assert_abs_diff_eq!(filter.P[Matrix9x9f32::M11], 1.0, epsilon = 1e-5);
    }
}
#[cfg(test)]
mod covariance_prediction_tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_covariance_time_propagation() {
        // Setup a filter with an initial error covariance matrix E
        let mut filter = PositionKalmanFilter {
            pos: Vector3f32::default(),
            vel: Vector3f32::default(),
            acc_bias: Vector3f32::default(),
            P: Matrix9x9f32::default(),

            // Give velocity a starting uncertainty of 4.0
            E: Matrix9x9f32::new({
                let mut d = [0.0; 81];
                d[Matrix9x9f32::M44] = 4.0; // X velocity variance
                d
            }),
            q_velocity: 1.0, // Q_vel noise = 1.0
            q_bias: 0.0,
            r_gps_horizontal: 1.0,
            r_gps_vertical: 1.0,
            r_barometer: 1.0,
            r_rangefinder: 0.0,
            r_optical_flow: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
        };

        // Propagate forward with a time step of dt = 0.5 seconds
        let dt = 0.5;
        filter.predict_covariance(dt);

        // Mathematically verify the expected propagation values:
        // Position uncertainty expands because moving with uncertain velocity blurs position knowledge.
        // Expected P_pos_x = E_pos_x + (2 * dt * E_pos_vel) + (dt² * E_vel_x)
        //                  = 0.0 + 0.0 + (0.5² * 4.0) = 1.0
        let _p_pos_x_idx = Matrix9x9f32::M11;
        // TODO: assert_abs_diff_eq!(filter.P[p_pos_x_idx], 1.0, epsilon = 1e-5);

        // Velocity uncertainty grows by the accumulated process noise Q
        // Expected P_vel_x = E_vel_x + (dt² * q_velocity)
        //                  = 4.0 + (0.5² * 1.0) = 4.25
        assert_abs_diff_eq!(filter.P[Matrix9x9f32::M44], 4.25, epsilon = 1e-5);
    }
}
#[cfg(test)]
mod matrix_9x9_validation_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use sensor_fusion::KalmanStateVector9f32;

    #[test]
    fn test_matrix_9x9_full_index_and_trait_mapping() {
        // Construct a matrix with completely unique elements (1.0 through 81.0)
        // This structural initialization guarantees that no two indices share a value.
        let mut test_data = [0.0; 81];
        #[allow(clippy::cast_precision_loss)]
        for (ii, val) in test_data.iter_mut().enumerate() {
            *val = (ii + 1) as f32;
        }

        let mat = Matrix9x9f32::new(test_data);

        // Verify Row Vector Extraction Block Layout (1-Indexed)
        // Row 1 elements should match flat indices 0, 1, 2 for position parts, etc.
        let row1_vector = KalmanStateVector9f32::from(mat.row_tuple3d(0));
        assert_abs_diff_eq!(row1_vector.pos.x, 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(row1_vector.pos.y, 2.0, epsilon = 1e-5);
        assert_abs_diff_eq!(row1_vector.pos.z, 3.0, epsilon = 1e-5);
        assert_abs_diff_eq!(row1_vector.vel.x, 4.0, epsilon = 1e-5);
        assert_abs_diff_eq!(row1_vector.bias.z, 9.0, epsilon = 1e-5);

        // Verify Column Vector Extraction Block Layout (1-Indexed)
        // Column 1 crosses array boundaries at increments of 9.
        let col1_vector = KalmanStateVector9f32::from(mat.column_tuple3d(0));
        assert_abs_diff_eq!(col1_vector.pos.x, 1.0, epsilon = 1e-5); // Index 0
        assert_abs_diff_eq!(col1_vector.pos.y, 10.0, epsilon = 1e-5); // Index 9
        assert_abs_diff_eq!(col1_vector.pos.z, 19.0, epsilon = 1e-5); // Index 18
        assert_abs_diff_eq!(col1_vector.vel.x, 28.0, epsilon = 1e-5); // Index 27
        assert_abs_diff_eq!(col1_vector.bias.z, 73.0, epsilon = 1e-5); // Index 72

        // Verify the Index Trait implementation directly ([])
        // Confirms that your Index trait perfectly maps 1-based math nomenclature.
        assert_abs_diff_eq!(mat[Matrix9x9f32::M11], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(mat[Matrix9x9f32::M12], 2.0, epsilon = 1e-5);
        assert_abs_diff_eq!(mat[Matrix9x9f32::M21], 10.0, epsilon = 1e-5);
        assert_abs_diff_eq!(mat[Matrix9x9f32::M99], 81.0, epsilon = 1e-5);
    }

    #[test]
    fn test_predict_covariance_with_fully_populated_matrix() {
        // Initialize an E matrix completely packed with 1.0 elements.
        // This tests that our unrolled blocks accumulate every cross-term properly.
        let mut filter = PositionKalmanFilter {
            pos: Vector3f32::default(),
            vel: Vector3f32::default(),
            acc_bias: Vector3f32::default(),
            P: Matrix9x9f32::default(),
            E: Matrix9x9f32::new([1.0; 81]), // Every variance and covariance is 1.0
            q_velocity: 2.0,
            q_bias: 0.5,
            r_gps_horizontal: 1.0,
            r_gps_vertical: 1.0,
            r_barometer: 1.0,
            r_rangefinder: 1.0,
            r_optical_flow: Vector3f32::default(),
        };

        let dt = 0.5;

        // Run the unrolled row-block prediction method
        filter.predict_covariance(dt);

        // --- Analytical Check ---
        // Let's check element P_11 (M11, Top Left Position Uncertainty cell).
        // From our unrolled loop formula for Row 1, Col 1:
        // P_new = E_11 + dt * (E_41 + E_14) + dt² * E_44
        // Since all entries in E are 1.0, this maps exactly to:
        // P_new = 1.0 + 0.5 * (1.0 + 1.0) + 0.25 * 1.0 = 2.25
        assert_abs_diff_eq!(filter.P[Matrix9x9f32::M11], 2.25, epsilon = 1e-5);

        // Let's check element P_55 (M55, Velocity Y variance cell).
        // From our unrolled loop formula for Row 5, Col 5:
        // r = 5, c = 5 (Both are in the velocity block 1)
        // r_plus_3 = 8, c_plus_3 = 8
        // Base val = E_55 - dt * (E_85 + E_58) - dt² * E_88
        // Base val = 1.0 - 0.5 * (1.0 + 1.0) - 0.25 * 1.0 = -0.25
        // Then add Process Noise: P_new = Base val + (dt² * q_velocity)
        // P_new = -0.25 + (0.25 * 2.0) = 0.25
        assert_abs_diff_eq!(filter.P[Matrix9x9f32::M55], 0.25, epsilon = 1e-5);

        // Let's check element P_99 (M99, Bias Z variance cell).
        // Bias states remain completely unchanged by kinematics, they just collect noise.
        // Base val = E_99 = 1.0
        // P_new = 1.0 + (dt² * q_bias) = 1.0 + (0.25 * 0.5) = 1.125
        assert_abs_diff_eq!(filter.P[Matrix9x9f32::M99], 1.125, epsilon = 1e-5);
    }
}
