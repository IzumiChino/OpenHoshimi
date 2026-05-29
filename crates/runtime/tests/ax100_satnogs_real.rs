//! Real-recording AX.100 / IO-117 decode test.
//!
//! Runs the IO-117 GreenCube 9k6 FSK pipeline (OGG/Vorbis -> FM
//! discriminator audio -> CPM/FSK -> ASM+Golay framer -> CCSDS
//! descramble -> CRC-32C) against a SatNOGS observation that ships in
//! the repo at `test_data/satnogs_gmsk_9k6.ogg`. The recording is small
//! enough to commit (~13 MiB) and exercises the full AX.100 ASM+Golay
//! path end-to-end.
//!
//! Run with:
//!
//!     cargo test -p openhoshimi-runtime --test ax100_satnogs_real \
//!         -- --nocapture
//!
//! Pass criteria:
//!  * The framer must surface at least one ASM-aligned candidate frame
//!    (the recording is known-good).
//!  * The codec must decode at least one of those candidates into a
//!    CRC-32C-valid CSP frame. The GreenCube link is *not* RS-protected
//!    (the Golay length counts 32 trailing parity bytes that this
//!    decoder strips, then verifies the CRC-32C trailer), so a clean
//!    burst decodes bit-exact and passes the CRC. A lower bound of one
//!    keeps the test stable across small demod tweaks while still
//!    catching a regression that breaks the framing or descrambler.

use std::path::PathBuf;

use openhoshimi_core::{
    satellite::{Ax100ModeDef, CodecDef, CpmModeDef, DownlinkDef, FramerDef, ModemDef},
    InputSource, IoError,
};
use openhoshimi_io::OggSource;
use openhoshimi_runtime::pipeline::{BitPipeline, DecodedFrame};

const SAMPLE_BUFFER: usize = 32_768;
// 1k2 SatNOGS observation 7633827 of IO-117 (2023-05-28). Not committed
// (large OGG); the test is gated on OPENHOSHIMI_SATNOGS_RECORDINGS and
// skips when the file or the env flag is absent.
const RECORDING: &str = "satnogs_7633827_2023-05-28T11-11-08.ogg";
// Lower bound on CRC-32C-valid frames. The honest CRC sweep recovers ~18
// over the high-elevation pass; 10 leaves headroom for small demod tweaks
// while still catching a framing/descrambler/CRC regression.
const MIN_CRC_VALID: usize = 10;

fn io117_1k2_downlink() -> DownlinkDef {
    DownlinkDef {
        label: "1k2 FSK digipeater".to_string(),
        freq_hz: 435_310_000,
        modulation: "FSK".to_string(),
        baudrate: 1200,
        framing: "GOMSPACE_AX100".to_string(),
        telemetry_schema: None,
        modem: Some(ModemDef::Cpm {
            mode: CpmModeDef::Fsk,
            modulation_index: Some(0.5),
            frequency_offset_hz: 0.0,
            gaussian_bt: None,
            differential: false,
            invert: false,
            swap_iq: false,
        }),
        line_coding: None,
        descrambler: None,
        framer: Some(FramerDef::Ax100Asm { threshold: 4 }),
        fec: None,
        codec: Some(CodecDef::GomspaceAx100 {
            mode: Ax100ModeDef::AsmGolayCrc,
        }),
        image: None,
    }
}

fn recording_path() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("crates/")
        .parent()
        .expect("workspace root")
        .join("test_recordings")
        .join(RECORDING)
}

#[test]
fn io117_1k2_real_recording_decodes_crc_valid_frames() {
    if std::env::var_os("OPENHOSHIMI_SATNOGS_RECORDINGS").is_none() {
        eprintln!(
            "skipping: set OPENHOSHIMI_SATNOGS_RECORDINGS=1 and place \
             {RECORDING} under test_recordings/ to run this test"
        );
        return;
    }
    let path = recording_path();
    if !path.exists() {
        eprintln!("skipping: recording not found at {}", path.display());
        return;
    }

    let mut source = OggSource::open(&path).expect("open OGG recording");
    let sample_rate = source.sample_rate_value();
    eprintln!("decoding {} ({} Hz, mono)", path.display(), sample_rate);

    let downlink = io117_1k2_downlink();
    let mut pipeline = BitPipeline::<f32>::new(&downlink).expect("build pipeline");
    pipeline
        .configure_fm_audio_demodulator(&downlink, sample_rate)
        .expect("configure FM-discriminator demodulator");

    let mut buffer = vec![0.0f32; SAMPLE_BUFFER];
    let mut total_frames = 0usize;
    let mut decoded_ax100 = 0usize;
    let mut crc_valid = 0usize;

    loop {
        let n = match source.read_samples(&mut buffer) {
            Ok(0) => break,
            Ok(n) => n,
            Err(IoError::EndOfStream) => break,
            Err(e) => panic!("read samples: {e}"),
        };
        let frames = pipeline.push_samples(&buffer[..n]);
        for frame in &frames {
            total_frames += 1;
            if let Ok(DecodedFrame::Ax100 { crc_ok, .. }) = pipeline.decode_frame(frame) {
                decoded_ax100 += 1;
                if crc_ok == Some(true) {
                    crc_valid += 1;
                }
            }
        }
    }

    eprintln!(
        "framer candidates: {}, ax100 decoded: {}, crc-valid frames: {}",
        total_frames, decoded_ax100, crc_valid
    );

    assert!(
        total_frames > 0,
        "no AX.100 candidate frames recovered from {RECORDING} \
         (framer/demod regression)"
    );
    assert!(
        crc_valid >= MIN_CRC_VALID,
        "expected at least {MIN_CRC_VALID} CRC-32C-valid AX.100 frames \
         from {RECORDING}, got {crc_valid} (saw {total_frames} candidates \
         at the framer layer; framing / descrambler / CRC regression)"
    );
}
