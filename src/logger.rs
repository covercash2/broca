use chrono::Local;
use fern::{self, Dispatch, colors::{Color, ColoredLevelConfig}};

pub fn init() {
    let colors_level = ColoredLevelConfig::new()
        .debug(Color::BrightCyan)
        .info(Color::BrightBlue);

    Dispatch::new()
	.format(move |out, message, record| {
	    out.finish(format_args!(
		"{date}[{level}][{target}] {message}",
		level = colors_level.color(record.level()),
		date = Local::now().format("[%y-%m-%d %H:%M:%S]"),
		target = record.target(),
		message = message
	    ))
	})
	.level(log::LevelFilter::Info)
        .level_for("broca", log::LevelFilter::Trace)
	.chain(std::io::stdout())
	.chain(fern::log_file("output.log").expect("could not open log file"))
	.apply().expect("unable to init logger");
}
