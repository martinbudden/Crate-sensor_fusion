# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

Releases of the form `0.1.n` do not adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html),
that is each release may contain incompatible API changes.

Once the API has stabilized this project will adopt semantic versioning, the first release to do so will be `0.2.0`.

## [Possible_future]

### Added

- May port  **VQF** (Versatile Quaternion Filter) from my C++ library [Library-SensorFusion](https://github.com/martinbudden/Library-SensorFusion).

## [Unreleased]

### Added

### Changed

### Removed

### Deprecated

### Fixed

### Security

## [0.1.9] - 2026-09-04

### Changed

- Use vqm version 0.1.14.

## [0.1.8] - 2026-07-04

### Added

- `KalmanStateVector9`.
- `trilaterate_2d` and `trilaterate_3d_weighted` functions.
- `#[must_use] attributes where appropriate.
- `deny`s in `lib.rs`.

### Changed

- Use vqm version 0.1.13.
- `AltitudeKalmanFilter` changed to use generics.

## [0.1.7] - 2026-06-08

### Added

- correct using rangefinder and using barometer functions to `AltitudeKalmanFilter`.
- correct using barometer and using gps to `PositionKalmanFilter`.
- `correct_yaw` to Madgwick filter.

### Changed

- Use vqm version 0.1.12.
- Changed AltitudeKalman filter to have predict and correct functions.
- Renamed update to correct for `PositionKalmanFilter`.
- Code tidy.

## [0.1.6] - 2026-05-24

### Changed

- Use vqm version 0.1.9.
- Improved efficiency of Kalman filters.
- Improved Kalman documentation.

## [0.1.5] - 2026-05-16

### Added

- `AltitudeKalmanFilter`
- `PositionKalmanFilter`

### Removed

- `katex-header.html`

## [0.1.4] - 2026-05-16

### Changed

- Use vqm version 0.1.5.

## [0.1.3] - 2026-05-06

### Changed

- Use vqm version 0.1.3.

## [0.1.2] - 2026-05-02

### Changed

- Improved README.md.
- Use vqm version 0.1.2.

## [0.1.1] - 2026-04-27

### Added

- `madgwick.rs` in `examples` directory.
- This changelog.
- CONTRIBUTING.md

### Changed

- Improved README.md.
- Improved documentation.
- Use vqm version 0.1.1.

## [0.1.0] - 2026-04-12

Initial release.
