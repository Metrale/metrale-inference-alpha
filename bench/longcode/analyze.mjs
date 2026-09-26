// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-26: Per-sample analyzer for the long-code harness: reads a sample JSON
// {seed, content, reasoning, finish_reason} (argv[2]) and prints metrics JSON.
//
// Owner: bench, long-code harness.
// Invariants: none beyond the types.
//
// Syntax validity comes from `node --check`; duplicate declarations from a walk
// of an acorn-loose AST.

import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const _req = createRequire(import.meta.url);
// 2026-09-26: acorn-loose parses the whole text, including a broken tail, so
// duplicates after a syntax error still count.
const acornLoose = _req("acorn-loose");

const sample = JSON.parse(readFileSync(process.argv[2], "utf8"));
const content = sample.content || "";
const finish = sample.finish_reason;

// 2026-09-26: fenceBlock takes the longest closed ``` block, an unclosed trailing
// fence (to the end of the text) when that is longer, or the whole text when it
// has no fence but looks like HTML or three.js. extractJs then keeps the inline
// <script> bodies, including an unclosed last one.
function fenceBlock(text) {
  let block = "";
  let lang = "";
  for (const m of text.matchAll(/```([a-zA-Z0-9]*)\n([\s\S]*?)```/g)) {
    if (m[2].length > block.length) {
      block = m[2];
      lang = (m[1] || "").toLowerCase();
    }
  }
  const open = text.match(/```([a-zA-Z0-9]*)\n([\s\S]*)$/);
  if (open && open[2].length > block.length && !open[2].includes("```")) {
    block = open[2];
    lang = (open[1] || "").toLowerCase();
  }
  if (!block && /<!DOCTYPE|<html|<script|new\s+THREE\./i.test(text)) {
    block = text;
  }
  return { block, lang };
}

function extractJs(text) {
  const { block, lang } = fenceBlock(text);
  if (!block) return "";
  const looksHtml = lang === "html" || /<script[\s>]/i.test(block);
  if (looksHtml) {
    const out = [];
    for (const s of block.matchAll(
      /<script\b([^>]*)>([\s\S]*?)<\/script(?=[\s/>])[^>]*>/gi
    )) {
      if (!/\bsrc\s*=/.test(s[1])) out.push(s[2]);
    }
    const lastOpen = block.match(/<script\b([^>]*)>([\s\S]*)$/i);
    if (
      lastOpen &&
      !/\bsrc\s*=/.test(lastOpen[1]) &&
      // 2026-09-26: The same end-tag pattern as the loop above, so a tail that
      // holds a closing tag (`</script >` included) is not taken as unclosed.
      !/<\/script(?=[\s\/>])[^>]*>/i.test(lastOpen[2])
    ) {
      out.push(lastOpen[2]);
    }
    if (out.length) return out.join("\n;\n");
  }
  return block;
}

const js = extractJs(content);
const tmp = mkdtempSync(join(tmpdir(), "lc-"));

// 2026-09-26: `node --check` on a JS string; true iff it parses.
function checks(src) {
  const f = join(tmp, "c.js");
  writeFileSync(f, src);
  try {
    execFileSync(process.execPath, ["--check", f], { stdio: "pipe" });
    return true;
  } catch {
    return false;
  }
}

// 2026-09-26: Longest line prefix that `node --check` accepts (binary search on lines).
function validLineCount(src) {
  const lines = src.split("\n");
  if (checks(src)) return lines.length;
  let lo = 0;
  let hi = lines.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (checks(lines.slice(0, mid).join("\n"))) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

// 2026-09-26: var and function names go to the nearest function (or program)
// scope, let and const to the nearest block; each repeat of a name in one bucket
// counts once. Class declarations are not counted.
function dupDeclCount(src) {
  const ast = acornLoose.parse(src, { ecmaVersion: "latest" });
  let dups = 0;
  let firstDupPos = null;
  const fnStack = [{ names: new Set() }];
  const blkStack = [{ names: new Set() }];

  function decl(name, kind, pos) {
    const bucket =
      kind === "var" || kind === "function"
        ? fnStack[fnStack.length - 1]
        : blkStack[blkStack.length - 1];
    if (bucket.names.has(name)) {
      dups += 1;
      if (firstDupPos === null) firstDupPos = pos;
    } else {
      bucket.names.add(name);
    }
  }

  function names(id, kind, pos) {
    if (!id) return;
    if (id.type === "Identifier") decl(id.name, kind, pos);
    else if (id.type === "ObjectPattern")
      id.properties.forEach((p) =>
        names(p.value || p.argument, kind, pos)
      );
    else if (id.type === "ArrayPattern")
      id.elements.forEach((e) => e && names(e, kind, pos));
    else if (id.type === "AssignmentPattern") names(id.left, kind, pos);
    else if (id.type === "RestElement") names(id.argument, kind, pos);
  }

  function walk(node) {
    if (!node || typeof node.type !== "string") return;
    const isFn =
      node.type === "FunctionDeclaration" ||
      node.type === "FunctionExpression" ||
      node.type === "ArrowFunctionExpression";
    const isBlk = node.type === "BlockStatement" || isFn;
    if (node.type === "VariableDeclaration")
      node.declarations.forEach((d) =>
        names(d.id, node.kind, d.start)
      );
    if (node.type === "FunctionDeclaration" && node.id)
      decl(node.id.name, "function", node.start);
    if (isFn) fnStack.push({ names: new Set() });
    if (isBlk) blkStack.push({ names: new Set() });
    for (const k of Object.keys(node)) {
      const v = node[k];
      if (Array.isArray(v)) v.forEach((c) => c && walk(c));
      else if (v && typeof v.type === "string") walk(v);
    }
    if (isBlk) blkStack.pop();
    if (isFn) fnStack.pop();
  }
  walk(ast);
  return { dups, firstDupPos };
}

const validLines = js ? validLineCount(js) : 0;
const totalLines = js ? js.split("\n").length : 0;
const { dups, firstDupPos } = js
  ? dupDeclCount(js)
  : { dups: 0, firstDupPos: null };

const badHex = [...content.matchAll(/0x(?![0-9a-fA-F])|0x[0-9a-fA-F]*[g-zG-Z]/g)];
const firstBadHex = badHex.length ? badHex[0].index : null;

// 2026-09-26: tokens_to_first_degeneration is a character offset into content, not a
// token count: the smallest of the first duplicate's position, the first malformed
// hex literal, and (after an unclean finish or without closed HTML) content.length.
const degCandidates = [];
if (firstDupPos !== null) {
  // 2026-09-26: firstDupPos indexes `js`; map it into content approximately via indexOf.
  const frag = js.slice(Math.max(0, firstDupPos - 12), firstDupPos + 12);
  const at = frag ? content.indexOf(frag.trim().split("\n")[0]) : -1;
  degCandidates.push(at >= 0 ? at : firstDupPos);
}
if (firstBadHex !== null) degCandidates.push(firstBadHex);
const closedHtml =
  content.includes("</script>") && content.includes("</html>");
if (finish !== "stop" || !closedHtml) degCandidates.push(content.length);
const tokensToFirstDegen = degCandidates.length
  ? Math.min(...degCandidates)
  : null;

const parsesClean = js ? checks(js) : false;
const hasScene = /new\s+THREE\.Scene\s*\(/.test(content);
const hasLoop =
  /requestAnimationFrame\s*\(/.test(content) ||
  /\.render\s*\(\s*scene/.test(content);
const completenessPass =
  parsesClean &&
  hasScene &&
  hasLoop &&
  closedHtml &&
  finish === "stop" &&
  dups === 0;

process.stdout.write(
  JSON.stringify({
    valid_js_line_count: validLines,
    total_js_line_count: totalLines,
    duplicate_declaration_count: dups,
    malformed_hex_count: badHex.length,
    tokens_to_first_degeneration: tokensToFirstDegen,
    parses_clean: parsesClean,
    has_scene: hasScene,
    has_render_loop: hasLoop,
    closed_html: closedHtml,
    completeness_pass: completenessPass,
  })
);
