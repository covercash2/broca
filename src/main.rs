//! Natural Language Processing functions and features
//! based on the rust-bert library
use async_std::sync::RecvError;
use log::*;

use async_std::{sync::{channel, Sender, Receiver}, task};

use tide::Request;
use tide::prelude::*;

use rust_bert::pipelines::generation_utils::{GPT2Generator, GenerateConfig, LanguageGenerator};
use rust_bert::RustBertError;
use tch::Device;

mod logger;

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
struct GenerateText {
    init_text: String,
    max_len: i64,
    min_len: i64,
}

#[derive(Debug)]
enum Error {
    Bert(RustBertError),
    SyncReceive(RecvError),
}

impl From<RustBertError> for Error {
    fn from(err: RustBertError) -> Self {
	Error::Bert(err)
    }
}

impl From<RecvError> for Error {
    fn from(err: RecvError) -> Self {
	Error::SyncReceive(err)
    }
}

async fn spawn_generator(receiver: Receiver<GenerateText>) -> Result<()>{
    let generate_config = GenerateConfig {
        max_length: 30,
        do_sample: true,
        num_beams: 5,
        temperature: 1.1,
        num_return_sequences: 3,
	device: Device::cuda_if_available(),
        ..Default::default()
    };
    let gpt2_generator = GPT2Generator::new(generate_config)?;

    debug!("gpt2 generator created");

    loop {
	debug!("awaiting input on receiver");
	let generate_text = receiver.recv().await?;
	debug!("input received: {:?}", generate_text);
	let min_length = Some(generate_text.min_len);
	let max_length = Some(generate_text.max_len);
	let decoder_start_id = None;

	let input_context = generate_text.init_text;

	let output = gpt2_generator.generate(
	    Some(vec![input_context.as_str()]),
	    None,
	    min_length,
	    max_length,
	    decoder_start_id,
	);
	debug!("output: {:?}", output);
    }
}

#[async_std::main]
async fn main() -> tide::Result<()> {
    logger::init();

    let (gpt_sender, gpt_receiver) = channel(16);

    task::spawn(spawn_generator(gpt_receiver));

    let generate_text = GenerateText {
	init_text: "The big bad bird".to_owned(),
	min_len: 32,
	max_len: 128,
    };

    gpt_sender.send(generate_text).await;

    let mut app = tide::new();
    app.listen("127.0.0.1:8080").await?;
    Ok(())
}
