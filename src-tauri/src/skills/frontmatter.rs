//! Reading the YAML block at the top of a `SKILL.md`, strictly.
//!
//! ## Why this is not a YAML library
//!
//! Two reasons, and the second is the one that decided it.
//!
//! The first is practical: this build resolves every Rust dependency from a
//! local registry with no network, and no YAML crate is vendored. Adding one is
//! not free here.
//!
//! The second is that **a skill file is untrusted input**. Full YAML is a large
//! grammar — anchors, aliases, merge keys, type tags, implicit typing — and
//! most of those features exist to let a document restructure itself as it is
//! read. A parser that supports them is a parser that can be surprised. This
//! one supports the smallest grammar that expresses a skill's metadata and
//! **refuses everything else by name**, so a file carrying an anchor is a
//! malformed skill rather than a clever one.
//!
//! What is accepted:
//!
//! ```text
//! name: inspection-approval-note        # scalar
//! description: >-                       # folded block scalar
//!   A sentence that runs
//!   across two lines.
//! allowed-tools:                        # sequence of scalars
//!   - search_documents
//!   - create_docx
//! compatibility:                        # one level of nesting, scalars only
//!   arjun: ">=0.1.0"
//!   requires-binaries: []
//! ```
//!
//! What is refused, explicitly and by name: anchors (`&`), aliases (`*`), tags
//! (`!`), merge keys (`<<`), flow mappings (`{...}`), non-empty flow sequences,
//! tabs, nesting past one level, and any line the grammar above does not
//! describe. A refusal names the line, because a skill author fixing one needs
//! to know which.

use std::collections::BTreeMap;

/// The largest frontmatter block that will be read.
///
/// Generous for metadata and small enough that a malformed file cannot be used
/// to make the registry read a large file into memory at discovery time, when
/// it is walking every skill directory on the machine.
pub const MAX_FRONTMATTER_BYTES: usize = 16 * 1024;

/// One value in the block. Deliberately three shapes and no more.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    Scalar(String),
    List(Vec<String>),
    /// One level of nesting, whose values are scalars or lists.
    Map(BTreeMap<String, Node>),
}

impl Node {
    pub fn as_scalar(&self) -> Option<&str> {
        match self {
            Node::Scalar(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[String]> {
        match self {
            Node::List(items) => Some(items),
            // A single scalar where a list was expected is the commonest
            // authoring slip, and reading it as a one-item list would hide it.
            _ => None,
        }
    }

    pub fn as_map(&self) -> Option<&BTreeMap<String, Node>> {
        match self {
            Node::Map(fields) => Some(fields),
            _ => None,
        }
    }
}

/// A parsed frontmatter block.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Document(pub BTreeMap<String, Node>);

impl Document {
    pub fn get(&self, key: &str) -> Option<&Node> {
        self.0.get(key)
    }

    pub fn scalar(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Node::as_scalar)
    }

    pub fn list(&self, key: &str) -> Option<&[String]> {
        self.get(key).and_then(Node::as_list)
    }

    pub fn map(&self, key: &str) -> Option<&BTreeMap<String, Node>> {
        self.get(key).and_then(Node::as_map)
    }
}

/// Why a block could not be read. Every variant names the line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// 1-based, counting from the start of the file.
    pub line: usize,
    pub problem: String,
}

impl ParseError {
    fn at(line: usize, problem: impl Into<String>) -> Self {
        Self {
            line,
            problem: problem.into(),
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.problem)
    }
}

/// The frontmatter block and the body that follows it.
#[derive(Debug, Clone)]
pub struct Split<'a> {
    pub frontmatter: &'a str,
    /// Everything after the closing `---`. Never read at discovery time.
    pub body: &'a str,
    /// Line the body starts on, so a later error can be reported honestly.
    pub body_line: usize,
}

/// Separates the frontmatter from the body without parsing either.
///
/// Kept apart from [`parse`] because discovery reads only the first half of the
/// file and the distinction is the whole of requirement 4: a registry that
/// splits and parses in one step is one that has already read every skill body
/// into memory before deciding whether it needed to.
pub fn split(source: &str) -> Result<Split<'_>, ParseError> {
    let mut lines = source.lines();
    let first = lines.next().ok_or_else(|| ParseError::at(1, "the file is empty"))?;
    if first.trim_end() != "---" {
        return Err(ParseError::at(
            1,
            "a SKILL.md must open with a line containing exactly `---`",
        ));
    }

    // Byte offset just past the opening delimiter.
    let mut offset = first.len() + 1;
    let mut line_number = 1;

    for line in lines {
        line_number += 1;
        if line.trim_end() == "---" {
            let frontmatter = &source[first.len() + 1..offset.min(source.len())];
            if frontmatter.len() > MAX_FRONTMATTER_BYTES {
                return Err(ParseError::at(
                    line_number,
                    format!(
                        "the frontmatter is {} bytes, above the {MAX_FRONTMATTER_BYTES} byte limit",
                        frontmatter.len()
                    ),
                ));
            }
            let body_start = (offset + line.len() + 1).min(source.len());
            return Ok(Split {
                frontmatter,
                body: &source[body_start..],
                body_line: line_number + 1,
            });
        }
        offset += line.len() + 1;
        if offset > MAX_FRONTMATTER_BYTES {
            return Err(ParseError::at(
                line_number,
                "the frontmatter block is longer than the limit, or its closing `---` is missing",
            ));
        }
    }

    Err(ParseError::at(
        line_number,
        "the frontmatter block was never closed with `---`",
    ))
}

/// Constructs that exist in YAML and are refused here, with the reason.
///
/// Checked before anything else on a line, so the message a skill author sees
/// names the feature rather than describing the parse failure it caused.
fn refuse_yaml_features(line: &str, number: usize) -> Result<(), ParseError> {
    let trimmed = line.trim_start();
    let refusal = |what: &str| {
        ParseError::at(
            number,
            format!(
                "{what} is not supported in a skill's frontmatter. Only plain keys, scalars, \
                 one level of nested keys, and lists of scalars are read."
            ),
        )
    };

    if line.contains('\t') {
        return Err(refusal("a tab character"));
    }
    if trimmed.starts_with("<<") {
        return Err(refusal("a merge key"));
    }
    // Anchors and aliases are only meaningful at the start of a value or item,
    // so a `&` inside prose is left alone.
    for (marker, what) in [('&', "an anchor"), ('*', "an alias"), ('!', "a tag")] {
        if value_of(trimmed).is_some_and(|value| value.starts_with(marker)) {
            return Err(refusal(what));
        }
        if let Some(item) = trimmed.strip_prefix("- ") {
            if item.trim_start().starts_with(marker) {
                return Err(refusal(what));
            }
        }
    }
    if value_of(trimmed).is_some_and(|value| value.starts_with('{')) {
        return Err(refusal("a flow mapping"));
    }
    Ok(())
}

/// The part after the first `:` on a `key: value` line, trimmed.
fn value_of(line: &str) -> Option<&str> {
    let (_, value) = line.split_once(": ").or_else(|| {
        line.strip_suffix(':').map(|key| (key, ""))
    })?;
    Some(value.trim())
}

/// How far a line is indented, in spaces.
fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// Reads a frontmatter block.
pub fn parse(frontmatter: &str) -> Result<Document, ParseError> {
    let lines: Vec<&str> = frontmatter.lines().collect();
    let mut fields: BTreeMap<String, Node> = BTreeMap::new();
    let mut index = 0;

    while index < lines.len() {
        // Offset by one for the opening `---`, so reported line numbers match
        // what an author sees in their editor.
        let number = index + 2;
        let line = lines[index];

        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            index += 1;
            continue;
        }
        refuse_yaml_features(line, number)?;

        if indent_of(line) != 0 {
            return Err(ParseError::at(
                number,
                "this line is indented but the line above it does not open a block",
            ));
        }

        let (key, rest) = split_key(line, number)?;
        if fields.contains_key(&key) {
            // Last-one-wins would let a crafted file put a permissive value
            // below a benign one and rely on the reader seeing the first.
            return Err(ParseError::at(number, format!("{key:?} is set twice")));
        }

        let (node, consumed) = read_value(&lines, index, rest, number)?;
        fields.insert(key, node);
        index += consumed;
    }

    Ok(Document(fields))
}

/// Splits `key: rest`, refusing anything that is not a plain key.
fn split_key(line: &str, number: usize) -> Result<(String, String), ParseError> {
    let trimmed = line.trim_end();
    let colon = trimmed.find(':').ok_or_else(|| {
        ParseError::at(
            number,
            "expected `key: value`, `key:` or `- item`, and this line is none of them",
        )
    })?;
    let key = trimmed[..colon].trim();
    if key.is_empty() {
        return Err(ParseError::at(number, "this line has no key before its `:`"));
    }
    if !key
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(ParseError::at(
            number,
            format!("{key:?} is not a plain key; letters, digits, `-` and `_` only"),
        ));
    }
    Ok((key.to_string(), trimmed[colon + 1..].trim().to_string()))
}

/// Reads the value belonging to the key on `lines[at]`.
///
/// Returns the node and how many lines it consumed, including the key's own.
fn read_value(
    lines: &[&str],
    at: usize,
    inline: String,
    number: usize,
) -> Result<(Node, usize), ParseError> {
    // `key: value` — the common case, and the only one that consumes one line.
    if !inline.is_empty() {
        if inline == "[]" {
            return Ok((Node::List(Vec::new()), 1));
        }
        if let Some(marker) = inline.chars().next().filter(|c| "|>".contains(*c)) {
            let fold = marker == '>';
            let chomp = inline.ends_with('-');
            let (text, consumed) = read_block_scalar(lines, at + 1, fold, chomp)?;
            return Ok((Node::Scalar(text), consumed + 1));
        }
        if inline.starts_with('[') {
            return Err(ParseError::at(
                number,
                "a flow sequence is only supported as `[]`; write a list on its own lines",
            ));
        }
        return Ok((Node::Scalar(unquote(&inline)), 1));
    }

    // `key:` with an indented block beneath it.
    let mut consumed = 1;
    let mut items: Vec<String> = Vec::new();
    let mut nested: BTreeMap<String, Node> = BTreeMap::new();

    while at + consumed < lines.len() {
        let line = lines[at + consumed];
        if line.trim().is_empty() {
            consumed += 1;
            continue;
        }
        let indent = indent_of(line);
        if indent == 0 {
            break;
        }
        let child_number = at + consumed + 2;
        refuse_yaml_features(line, child_number)?;

        let trimmed = line.trim_start();
        if let Some(item) = trimmed.strip_prefix("- ") {
            if !nested.is_empty() {
                return Err(ParseError::at(
                    child_number,
                    "this block mixes list items and keys, which is ambiguous",
                ));
            }
            items.push(unquote(item.trim()));
            consumed += 1;
            continue;
        }

        if !items.is_empty() {
            return Err(ParseError::at(
                child_number,
                "this block mixes list items and keys, which is ambiguous",
            ));
        }

        let (key, rest) = split_key(trimmed, child_number)?;
        if nested.contains_key(&key) {
            return Err(ParseError::at(child_number, format!("{key:?} is set twice")));
        }
        if rest.is_empty() {
            // A nested key with a block under it. The one shape supported is a
            // list of scalars — `compatibility.requires-binaries` needs it, and
            // it is the only nested block a skill's metadata has a use for.
            // A nested *map* is refused: depth is where a reader's assumptions
            // and an author's diverge, and two levels is enough.
            let (items, used) = read_nested_list(lines, at + consumed + 1, indent)?;
            if items.is_empty() && used == 0 {
                nested.insert(key, Node::Scalar(String::new()));
                consumed += 1;
                continue;
            }
            nested.insert(key, Node::List(items));
            consumed += 1 + used;
            continue;
        }
        if rest == "[]" {
            nested.insert(key, Node::List(Vec::new()));
            consumed += 1;
            continue;
        }
        nested.insert(key, Node::Scalar(unquote(&rest)));
        consumed += 1;
    }

    if !items.is_empty() {
        return Ok((Node::List(items), consumed));
    }
    if !nested.is_empty() {
        return Ok((Node::Map(nested), consumed));
    }
    // `key:` with nothing under it is an empty scalar, not an error: an author
    // clearing a field should not have to delete the line.
    Ok((Node::Scalar(String::new()), consumed))
}

/// Reads a list of scalars indented under a nested key.
///
/// Returns the items and how many lines they took. `(empty, 0)` means there was
/// no block at all, which is a nested key with an empty value rather than an
/// error — an author clearing a field should not have to delete the line.
fn read_nested_list(
    lines: &[&str],
    from: usize,
    parent_indent: usize,
) -> Result<(Vec<String>, usize), ParseError> {
    let mut items = Vec::new();
    let mut consumed = 0;

    while from + consumed < lines.len() {
        let line = lines[from + consumed];
        if line.trim().is_empty() {
            // A blank line inside the block is skipped; one after it ends the
            // block only if what follows is less indented, which the checks
            // below decide.
            consumed += 1;
            continue;
        }
        if indent_of(line) <= parent_indent {
            break;
        }
        let number = from + consumed + 2;
        refuse_yaml_features(line, number)?;

        let trimmed = line.trim_start();
        let Some(item) = trimmed.strip_prefix("- ") else {
            return Err(ParseError::at(
                number,
                "frontmatter nests one level only, and a nested key may hold a list of scalars",
            ));
        };
        items.push(unquote(item.trim()));
        consumed += 1;
    }

    // Trailing blank lines were counted while looking ahead; give them back so
    // the caller resumes on the right line.
    while consumed > 0 && lines[from + consumed - 1].trim().is_empty() {
        consumed -= 1;
    }
    Ok((items, consumed))
}

/// Reads an indented block scalar introduced by `|` or `>`.
fn read_block_scalar(
    lines: &[&str],
    from: usize,
    fold: bool,
    chomp: bool,
) -> Result<(String, usize), ParseError> {
    let mut collected: Vec<String> = Vec::new();
    let mut consumed = 0;

    while from + consumed < lines.len() {
        let line = lines[from + consumed];
        if line.trim().is_empty() {
            collected.push(String::new());
            consumed += 1;
            continue;
        }
        if indent_of(line) == 0 {
            break;
        }
        if line.contains('\t') {
            return Err(ParseError::at(
                from + consumed + 2,
                "a tab character is not supported in a skill's frontmatter",
            ));
        }
        collected.push(line.trim_start().to_string());
        consumed += 1;
    }

    while collected.last().is_some_and(|line| line.is_empty()) {
        collected.pop();
    }

    let mut text = if fold {
        // Folded: single newlines become spaces, blank lines stay as breaks.
        let mut out = String::new();
        for line in &collected {
            if line.is_empty() {
                out.push('\n');
            } else {
                if !out.is_empty() && !out.ends_with('\n') {
                    out.push(' ');
                }
                out.push_str(line);
            }
        }
        out
    } else {
        collected.join("\n")
    };

    if !chomp && !text.is_empty() {
        text.push('\n');
    }
    Ok((text, consumed))
}

/// Removes one layer of matching quotes, without interpreting escapes.
///
/// Deliberately no escape handling: a skill's metadata is names, versions and
/// sentences, and an escape grammar is one more thing a crafted file could
/// exploit. A value that needs a quote inside it can use a block scalar.
fn unquote(value: &str) -> String {
    let trimmed = value.trim();
    for quote in ['"', '\''] {
        if trimmed.len() >= 2 && trimmed.starts_with(quote) && trimmed.ends_with(quote) {
            return trimmed[1..trimmed.len() - 1].to_string();
        }
    }
    trimmed.to_string()
}
