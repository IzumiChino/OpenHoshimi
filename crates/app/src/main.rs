//! egui desktop application for OpenHoshimi.
//!
//! Minimal desktop shell while the decoder pipeline is built out.

#![forbid(unsafe_code)]

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_simple_native("OpenHoshimi", options, |ctx, _frame| {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("OpenHoshimi");
            ui.label("Decoder workspace stub.");
        });
    })
}
