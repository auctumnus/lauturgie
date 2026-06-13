fn main() {
    for name in [
        "empty",
        "kharulian",
        "muipidan",
        "nitherwe",
        "syllabian",
        "unicode",
    ] {
        let path = format!("vendor/lexurgy/cli/test/{name}.lsc");
        let Ok(src) = std::fs::read_to_string(&path) else {
            println!("{name}: (missing)");
            continue;
        };
        match lauturgie::parse(&src) {
            Ok(stmts) => match lauturgie::compiler::compile(&stmts) {
                Ok(c) => println!(
                    "{name}: OK, {} rules, {} segments interned",
                    c.rules.len(),
                    c.segments.len()
                ),
                Err(e) => println!("{name}: compile error: {e}"),
            },
            Err(e) => println!("{name}: parse error: {e}"),
        }
    }
}
