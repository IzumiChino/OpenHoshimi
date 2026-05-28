//! Real-recording AX.100 / IO-117 decode test.
//!
//! Runs the IO-117 GreenCube 9k6 FSK pipeline (OGG/Vorbis -> FM
//! discriminator audio -> CPM/FSK -> ASM+Golay framer -> RS(255,223))
//! against a SatNOGS observation that ships in the repo at
//! `test_data/satnogs_gmsk_9k6.ogg`. The recording is small enough to
//! commit (~13 MiB) and exercises the full hard- and soft-decision
//! AX.100 paths end-to-end.
//!
//! Run with:
//!
//!     cargo test -p openhoshimi-runtime --test ax100_satnogs_real \
//!         -- --nocapture
//!
//! Pass criteria:
//!  * The framer must surface at least one ASM-aligned candidate frame
//!    (the recording is known-good and has produced 12 in the
//!    decode_file sweep).
//!  * The codec must decode at least one of those frames into a valid
//!    RS(255,223) payload — the soft-decision erasure path with K=32
//!    recovers 5 of the 12 candidates on this recording, so a lower
//!    bound of one keeps the test stable across small pipeline tweaks.

use std::path::PathBuf;

use openhoshimi_core::{
    satellite::{Ax100ModeDef, CodecDef, CpmModeDef, DownlinkDef, FramerDef, ModemDef},
    InputSource, IoError,
};
use openhoshimi_io::OggSource;
use openhoshimi_runtime::pipeline::{BitPipeline, DecodedFrame};

const SAMPLE_BUFFER: usize = 32_768;
const RECORDING: &str = "satnogs_gmsk_9k6.ogg";

fn io117_9k6_downlink() -> DownlinkDef {
    DownlinkDef {
        label: "9k6 FSK digipeater".to_string(),
        freq_hz: 435_310_000,
        modulation: "FSK".to_string(),
        baudrate: 9600,
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
            mode: Ax100ModeDef::AsmGolay,
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
        .join("test_data")
        .join(RECORDING)
}

#[test]
fn io117_9k6_real_recording_decodes_at_least_one_frame() {
    let path = recording_path();
    assert!(
        path.exists(),
        "expected committed recording at {}",
        path.display()
    );

    let mut source = OggSource::open(&path).expect("open OGG recording");
    let sample_rate = source.sample_rate_value();
    eprintln!("decoding {} ({} Hz, mono)", path.display(), sample_rate);

    let downlink = io117_9k6_downlink();
    let mut pipeline = BitPipeline::<f32>::new(&downlink).expect("build pipeline");
    pipeline
        .configure_fm_audio_demodulator(&downlink, sample_rate)
        .expect("configure FM-discriminator demodulator");

    let mut buffer = vec![0.0f32; SAMPLE_BUFFER];
    let mut total_frames = 0usize;
    let mut decoded_ax100 = 0usize;
    let mut total_corrected = 0usize;

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
            if let Ok(DecodedFrame::Ax100 {
                corrected_errors, ..
            }) = pipeline.decode_frame(frame)
            {
                decoded_ax100 += 1;
                total_corrected += corrected_errors;
            }
        }
    }

    eprintln!(
        "framer candidates: {}, ax100 decoded: {}, total corrected bytes: {}",
        total_frames, decoded_ax100, total_corrected
    );

    assert!(
        total_frames > 0,
        "no AX.100 candidate frames recovered from {RECORDING} \
         (framer/demod regression)"
    );
    assert!(
        decoded_ax100 > 0,
        "no AX.100 RS payload decoded from {RECORDING} \
         (saw {total_frames} candidates at the framer layer; codec / soft-erasure regression)"
    );
}
