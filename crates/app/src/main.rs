//! egui desktop application for OpenHoshimi.
//!
//! Stub for Phase 0/1. Real UI lands in Phase 5.

#![forbid(unsafe_code)]

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_simple_native("OpenHoshimi", options, |ctx, _frame| {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("OpenHoshimi");
            ui.label("UI stub - see Phase 5.");
        });
    })
}
