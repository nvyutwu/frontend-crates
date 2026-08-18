// Why the tear fires on DSpark but not on the single-token arms.
//
// The earlier repro chunked by fixed character width, which splits mid-token — something a real
// backend never does. This one chunks on TOKEN boundaries (K3 structural markers are single special
// tokens; words are text tokens) and varies only K = tokens per delta. K=1 models baseline/DCP,
// K=4 models DSpark (3 accepted draft tokens + 1 bonus).
use dynamo_parsers::reasoning::{ReasoningParser, ReasoningParserType};

/// One decode step's worth of text = K whole tokens joined.
fn deltas(tokens: &[&str], k: usize) -> Vec<String> {
    tokens.chunks(k).map(|c| c.concat()).collect()
}

fn run(tokens: &[&str], k: usize) -> (String, String) {
    let mut p = ReasoningParserType::KimiK3.get_reasoning_parser();
    p.set_in_reasoning(true);
    let (mut r, mut n) = (String::new(), String::new());
    for d in deltas(tokens, k) {
        let out = p.parse_reasoning_streaming_incremental(&d, &[]);
        r.push_str(&out.reasoning_text);
        n.push_str(&out.normal_text);
    }
    let out = p.finish_reasoning_stream();
    r.push_str(&out.reasoning_text);
    n.push_str(&out.normal_text);
    (r, n)
}

#[test]
fn token_aligned_k1_vs_k4() {
    // A K3 tool-call completion as the tokenizer actually emits it: structural markers are their
    // own special tokens, prose is text tokens.
    let tokens: &[&str] = &[
        "Need", " the", " weather", " tool", ".",
        "<|close|>", "think", "<|sep|>",
        "<|open|>", "tools", "<|sep|>",
        "<|open|>", "call", " tool=\"get_weather\"", "<|sep|>",
        "<|open|>", "argument", " key=\"city\"", "<|sep|>", "Paris", "<|close|>", "argument", "<|sep|>",
        "<|close|>", "call", "<|sep|>",
        "<|close|>", "tools", "<|sep|>",
        "<|close|>", "message", "<|sep|>", "<|end_of_msg|>",
    ];
    let expect_reasoning = "Need the weather tool.";
    let expect_normal: String = tokens
        .iter()
        .skip_while(|t| **t != "<|close|>")
        .skip(3) // drop the `<|close|>think<|sep|>` triple
        .copied()
        .collect();

    for k in 1..=6usize {
        let (r, n) = run(tokens, k);
        println!("K={k}\n   R={r:?}\n   N_ok={}", n == expect_normal);
        if n != expect_normal {
            println!("   N={n:?}");
        }
        assert_eq!(r, expect_reasoning, "K={k}: reasoning polluted");
        assert_eq!(n, expect_normal, "K={k}: tool section corrupted");
    }
}
