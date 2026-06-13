//! Scratch: find which rule first makes a word error by compiling growing
//! prefixes of the statement list.
//!
//! Usage: bisect_error CHANGES.lsc WORD

fn main() {
    let mut args = std::env::args().skip(1);
    let lsc = args.next().unwrap();
    let word = args.next().unwrap();
    let src = std::fs::read_to_string(&lsc).unwrap();
    let statements = lauturgie::parse(&src).unwrap();

    let name = |s: &lauturgie::parser::ast::Statement| -> String {
        use lauturgie::parser::ast::Statement::*;
        match s {
            Rule(r) => format!("rule {}", r.name),
            Syllables(_) => "syllables".into(),
            Romanizer { .. } => "romanizer".into(),
            InterRomanizer { name, .. } => format!("romanizer-{name}"),
            Deromanizer { .. } => "deromanizer".into(),
            _ => "decl".into(),
        }
    };

    let mut prev: Option<String> = None;
    for n in 1..=statements.len() {
        let mut compiled = match lauturgie::compiler::compile(&statements[..n]) {
            Ok(c) => c,
            Err(_) => continue, // prefix not yet self-contained
        };
        match compiled.apply(&word) {
            Ok(o) => prev = Some(o),
            Err(e) => {
                println!(
                    "errors at statement {n} ({}): {e:?}\n  word before: {}",
                    name(&statements[n - 1]),
                    prev.as_deref().unwrap_or("?")
                );
                return;
            }
        }
    }
    println!("no error: {}", prev.unwrap_or_default());
}
