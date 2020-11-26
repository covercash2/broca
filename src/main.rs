//! Natural Language Processing functions and features
//! based on the rust-bert library
use log::*;

use rust_bert::pipelines::generation_utils::{GPT2Generator, GenerateConfig, LanguageGenerator};
use rust_bert::RustBertError;

mod logger;

fn run_generator() -> Result<(), RustBertError> {
    let generate_config = GenerateConfig {
        max_length: 30,
        do_sample: true,
        num_beams: 5,
        temperature: 1.1,
        num_return_sequences: 3,
        ..Default::default()
    };
    let gpt2_generator = GPT2Generator::new(generate_config)?;

    let min_length = Some(32);
    let max_length = Some(64);
    let decoder_start_id = None;

    let input_context = "The dog";
    //let second_input_context = "The cat was";
    let output = gpt2_generator.generate(
        //Some(vec![input_context, second_input_context]),
        Some(vec![input_context]),
        None,
        min_length,
        max_length,
        decoder_start_id,
    );

    debug!("output: {:?}", output);

    Ok(())
}

pub fn main() {
    logger::init();

    run_generator().expect("unable to run gpt text generator");
}
