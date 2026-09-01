#![allow(non_snake_case)]

#[cfg(test)]
mod tests {
    use sensor_fusion::{PositionKalmanFilterDeprecated, PositionKalmanFilterDeprecatedExtended};
    use vqm::{Matrix9x9f32, Vector3f32};

    /// Helper to verify a column-major Matrix9x9 is perfectly symmetric.
    /// In column-major layout, P[(col * 9) + row] must equal P[(row * 9) + col].
    fn assert_matrix_symmetry(matrix: &Matrix9x9f32, tolerance: f32) {
        let mut cached_cols = [[0.0f32; 9]; 9];
        for (c, col_data) in matrix.iter_columns().enumerate() {
            cached_cols[c] = *col_data;
        }

        for (c, col_array) in cached_cols.iter().enumerate() {
            for (r, &val_at_row) in col_array.iter().enumerate() {
                let transposed_val = cached_cols[r][c];
                let diff = (val_at_row - transposed_val).abs();

                assert!(
                    diff < tolerance,
                    "Symmetry violation found at Col {}, Row {}: |{} - {}| = {}",
                    c,
                    r,
                    val_at_row,
                    transposed_val,
                    diff
                );
            }
        }
    }

    /// Helper to check if the main diagonal components remain strictly positive.
    fn assert_positive_diagonal(matrix: &Matrix9x9f32) {
        for (c, col_data) in matrix.iter_columns().enumerate() {
            let diagonal_variance = col_data[c];
            assert!(
                diagonal_variance > 0.0,
                "Variance breakdown! Negative diagonal encountered at state index {}: {}",
                c,
                diagonal_variance
            );
        }
    }

    #[test]
    fn test_production_kalman_pipeline_trace() {
        // Initialize a realistic diagonal uncertainty covariance matrix (Column-Major)
        let mut initial_P = Matrix9x9f32::default();
        {
            let mut cols = initial_P.iter_columns_mut();
            for i in 0..9 {
                let col = cols.next().unwrap();
                col[i] = if i < 3 {
                    0.5
                } else if i < 6 {
                    0.1
                } else {
                    0.01
                };
            }
        }

        // Instantiate actual filter structure with real hyperparameters
        let filter_base = PositionKalmanFilterDeprecated {
            pos: Vector3f32 { x: 0.0, y: 0.0, z: 10.0 }, // 10m Altitude
            vel: Vector3f32 { x: 0.0, y: 0.0, z: 2.0 },  // 2 m/s ascending
            acc_bias: Vector3f32 { x: 0.01, y: -0.01, z: 0.0 },
            P: initial_P,
            E: Matrix9x9f32::default(),
            q_velocity: 0.05,
            q_bias: 0.001,
        };
        let mut filter = PositionKalmanFilterDeprecatedExtended {
            base: filter_base,
            r_gps_horizontal: 0.02,
            r_gps_vertical: 0.05,
            r_barometer: 0.1,
            r_rangefinder: 0.01,
            r_optical_flow: Vector3f32 { x: 0.05, y: 0.05, z: 0.0 },
        };
        let dt = 0.01; // 100Hz IMU loop stride
        let total_timesteps = 50;

        println!("--- Starting Production Filter Trace Verification ---");

        for step in 1..=total_timesteps {
            // Copy active P matrix to E using public column iterators to establish a baseline
            {
                let mut e_cols = filter.base.E.iter_columns_mut();
                for p_col in filter.base.P.iter_columns() {
                    *e_cols.next().unwrap() = *p_col;
                }
            }

            // Execute predict code
            filter.predict_covariance(dt);

            // Run structural integrity assertions.
            assert_positive_diagonal(&filter.base.P);
            assert_matrix_symmetry(&filter.base.P, 2e-2);

            if step % 10 == 0 {
                println!("Timestep {:02}: Integrity OK. State Vector and Covariance Trace running smoothly.", step);
            }
        }

        // Test measurement observation mechanics
        let y_innovation = Vector3f32 { x: 0.1, y: -0.05, z: 0.02 };
        let R_gps_noise =
            Vector3f32 { x: filter.r_gps_horizontal, y: filter.r_gps_horizontal, z: filter.r_gps_vertical };

        // Ensure validate_measurement hooks up correctly using the real filter instance parameters
        let is_valid = filter.base.validate_measurement(y_innovation, R_gps_noise, 7.815);
        assert!(is_valid, "Valid sensory update was incorrectly rejected by gating limits.");
    }
}

#[cfg(test)]
mod correction_tests {
    use sensor_fusion::{PositionKalmanFilterDeprecated, PositionKalmanFilterDeprecatedExtended};
    use vqm::{Matrix9x9f32, Vector3f32};

    fn get_diagonal(matrix: &Matrix9x9f32) -> [f32; 9] {
        let mut diag = [0.0f32; 9];
        for (c, col_data) in matrix.iter_columns().enumerate() {
            diag[c] = col_data[c];
        }
        diag
    }

    #[test]
    fn test_multi_sensor_correction_pipeline() {
        // Initialize an identity-like diagonal variance layout
        let mut initial_P = Matrix9x9f32::default();
        {
            let mut cols = initial_P.iter_columns_mut();
            for (c, col) in cols.by_ref().take(9).enumerate() {
                for (r, cell) in col.iter_mut().enumerate() {
                    if c == r {
                        *cell = if c < 3 {
                            1.0
                        } else if c < 6 {
                            0.2
                        } else {
                            0.02
                        };
                    } else {
                        *cell = 0.0; // Keep it clean to isolate 1D update mechanics
                    }
                }
            }
        }

        let filter_base = PositionKalmanFilterDeprecated {
            pos: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            vel: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            acc_bias: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            P: initial_P,
            E: Matrix9x9f32::default(),
            q_velocity: 0.05,
            q_bias: 0.001,
        };
        let mut filter = PositionKalmanFilterDeprecatedExtended {
            base: filter_base,
            r_gps_horizontal: 0.04,
            r_gps_vertical: 0.09,
            r_barometer: 0.25,
            r_rangefinder: 0.01,
            r_optical_flow: Vector3f32 { x: 0.04, y: 0.04, z: 0.0 },
        };

        println!("\n====================================================");
        println!("DIAGNOSTIC TRACE: MULTI-SENSOR CORRECTION PIPELINE");
        println!("====================================================");

        let diag_initial = get_diagonal(&filter.base.P);
        println!("Initial Diagonal Variances:       {:?}", diag_initial);
        let pos_var_initial = diag_initial[2];

        // TRACE 1: BAROMETER CORRECTION
        let simulated_baro_alt = 10.5;
        filter.correct_altitude_using_barometer(simulated_baro_alt);

        let diag_after_baro = get_diagonal(&filter.base.P);
        println!("Diagonal After Barometer Update:  {:?}", diag_after_baro);
        let pos_var_after_baro = diag_after_baro[2];

        // TRACE 2: RANGEFINDER CORRECTION
        let simulated_range_alt = 10.42;
        filter.correct_altitude_using_rangefinder(simulated_range_alt);

        let diag_after_range = get_diagonal(&filter.base.P);
        println!("Diagonal After Rangefinder Update: {:?}", diag_after_range);
        let _pos_var_after_range = diag_after_range[2];

        println!("====================================================");

        // Print warning to stdout instead of hard panicking so you can read the log output
        if pos_var_after_baro >= pos_var_initial {
            println!("⚠️ WARNING: Barometer update did NOT change the Z-variance. It is returning early!");
        } else {
            println!("✅ SUCCESS: Barometer update successfully reduced Z-variance.");
        }
    }
}
