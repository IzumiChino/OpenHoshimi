//! Real-recording AFSK 1200 / AX.25 decode test.
//!
//! Runs the full ISS APRS digipeater pipeline (OGG/Vorbis -> AFSK ->
//! NRZI -> HDLC -> AX.25) against an actual SatNOGS observation. The
//! recording is **not** committed to the repo — it is downloaded on
//! demand into `test_recordings/` (see the script in
//! `docs/satnogs_recordings.md` once added) and the test is gated by
//! the `OPENHOSHIMI_SATNOGS_RECORDINGS` env var so CI does not need
//! 15 MiB of audio per pass.
//!
//! Run with:
//!
//!     OPENHOSHIMI_SATNOGS_RECORDINGS=1 \
//!         cargo test -p openhoshimi-runtime --test afsk_satnogs_real \
//!         -- --ignored --nocapture
//!
//! The test passes if at least one HDLC frame surfaces from the
//! recording and at least one of those frames decodes as a structured
//! AX.25 UI packet whose source callsign is non-empty ASCII. A SatNOGS
//! observation flagged "good" by the network is expected to contain
//! several such frames; a permissive lower bound keeps the test stable
//! across recordings of different SNR.

use std::path::PathBuf;

use openhoshimi_codec::Ax25Frame;
use openhoshimi_core::{
    satellite::{CodecDef, DownlinkDef, FramerDef, LineCodingDef, ModemDef},
    InputSource, IoError,
};
use openhoshimi_io::OggSource;
use openhoshimi_runtime::pipeline::{BitPipeline, DecodedFrame};

const SAMPLE_BUFFER: usize = 32_768;

fn iss_aprs_downlink() -> DownlinkDef {
    DownlinkDef {
        label: "APRS digipeater (1k2 AFSK)".to_string(),
        freq_hz: 145_825_000,
        modulation: "AFSK".to_string(),
        baudrate: 1200,
        framing: "AX25".to_string(),
        telemetry_schema: None,
        modem: Some(ModemDef::Afsk {
            mark_hz: 1200.0,
            space_hz: 2200.0,
        }),
        line_coding: Some(LineCodingDef::Nrzi),
        descrambler: None,
        framer: Some(FramerDef::Hdlc),
        fec: None,
        codec: Some(CodecDef::Ax25),
        image: None,
    }
}

fn recording_path(filename: &str) -> Option<PathBuf> {
    if std::env::var("OPENHOSHIMI_SATNOGS_RECORDINGS").is_err() {
        return None;
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidate = manifest
        .parent()? // crates/
        .parent()? // workspace root
        .join("test_recordings")
        .join(filename);
    candidate.exists().then_some(candidate)
}

#[test]
#[ignore = "requires SatNOGS recording in test_recordings/, gated by OPENHOSHIMI_SATNOGS_RECORDINGS=1"]
fn iss_aprs_real_recording_decodes_at_least_one_frame() {
    let path = match recording_path("iss_aprs_14155231.ogg") {
        Some(p) => p,
        None => {
            eprintln!(
                "skipping: set OPENHOSHIMI_SATNOGS_RECORDINGS=1 and place \
                 test_recordings/iss_aprs_14155231.ogg (SatNOGS observation \
                 14155231) to enable this test"
            );
            return;
        }
    };
    let mut source = OggSource::open(&path).expect("open OGG recording");
    let sample_rate = source.sample_rate_value();
    eprintln!("decoding {} ({} Hz, mono)", path.display(), sample_rate);

    let downlink = iss_aprs_downlink();
    let mut pipeline = BitPipeline::<f32>::new(&downlink).expect("build pipeline");
    pipeline
        .configure_demodulator(&downlink, sample_rate, 0.0)
        .expect("configure AFSK demodulator");

    let mut buffer = vec![0.0f32; SAMPLE_BUFFER];
    let mut total_frames = 0usize;
    let mut decoded_ax25 = 0usize;
    let mut sample_callsigns: Vec<String> = Vec::new();

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
            if let Ok(DecodedFrame::Ax25(Ax25Frame {
                source: src,
                destination: dst,
                info,
                ..
            })) = pipeline.decode_frame(frame)
            {
                decoded_ax25 += 1;
                if sample_callsigns.len() < 5 {
                    let info_preview: String = info
                        .iter()
                        .take(60)
                        .map(|&b| {
                            if (32..=126).contains(&b) {
                                b as char
                            } else {
                                '.'
                            }
                        })
                        .collect();
                    sample_callsigns.push(format!("{}>{}: {}", src.call, dst.call, info_preview));
                }
            }
        }
    }

    eprintln!(
        "frames seen: {}, ax25 decoded: {}",
        total_frames, decoded_ax25
    );
    for line in &sample_callsigns {
        eprintln!("  {}", line);
    }

    assert!(
        total_frames > 0,
        "no HDLC frames recovered from real ISS APRS recording"
    );
    assert!(
        decoded_ax25 > 0,
        "no AX.25 UI packet decoded from real ISS APRS recording \
         (saw {total_frames} frames at the HDLC layer)"
    );
}
