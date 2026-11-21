// Quarto post-render step: re-highlight the code blocks of the rendered pages
// with Shiki, at build time.
//
// Only `<code class="sourceCode LANG">` blocks are touched, that is, source code
// (cells, fenced blocks, signatures). Executed outputs are plain `<pre><code>`
// and stay as they are. Quarto's line spans (`<span id="cbN-k"><a ...></a>`)
// are kept, so code copy and line anchors still work.
//
// Each token carries both colours as CSS variables (`--shiki-light`,
// `--shiki-dark`); the theme stylesheet of the active mode picks one.
//
// Files: the paths in QUARTO_PROJECT_OUTPUT_FILES (or in the file named by
// QUARTO_USE_FILE_FOR_PROJECT_OUTPUT_FILES) when Quarto runs the script, else
// the paths given on the command line.

import { readFileSync, writeFileSync } from "node:fs";
import { createHighlighter } from "shiki";

const LANGS = ["python", "bash", "bibtex", "diff", "ini", "toml", "yaml"];
const THEMES = {
  light: JSON.parse(readFileSync(new URL("./laplacekit-light.json", import.meta.url), "utf8")),
  dark: JSON.parse(readFileSync(new URL("./laplacekit-dark.json", import.meta.url), "utf8")),
};

const BLOCK = /<code class="sourceCode ([\w-]+)"[^>]*>([\s\S]*?)<\/code>/g;
const LINE = /^<span id="([\w-]+)"><a href="#\1" aria-hidden="true" tabindex="-1"><\/a>([\s\S]*)<\/span>$/;

const ENTITIES = { "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'" };

function unescape(html) {
  return html
    .replace(/<[^>]*>/g, "")
    .replace(/&(amp|lt|gt|quot|#39);/g, (e) => ENTITIES[e])
    .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(Number(n)))
    .replace(/&#x([0-9a-f]+);/gi, (_, n) => String.fromCodePoint(parseInt(n, 16)));
}

function escape(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function style(vars) {
  return Object.entries(vars)
    .map(([key, value]) => `${key}:${value}`)
    .join(";");
}

/** The highlighted inner HTML of one block, or null when its lines do not
 * have Quarto's shape. */
function highlightBlock(highlighter, lang, inner) {
  const lines = inner.split("\n").map((line) => line.match(LINE));

  if (lines.some((line) => line === null)) {
    return null;
  }

  const code = lines.map((line) => unescape(line[2])).join("\n");
  const { tokens, rootStyle } = highlighter.codeToTokens(code, {
    lang,
    themes: { light: "laplacekit-light", dark: "laplacekit-dark" },
    defaultColor: false,
  });

  const html = tokens.map((lineTokens, i) => {
    const id = lines[i][1];
    const body = lineTokens
      .map((token) => `<span style="${style(token.htmlStyle ?? {})}">${escape(token.content)}</span>`)
      .join("");

    return `<span id="${id}"><a href="#${id}" aria-hidden="true" tabindex="-1"></a>${body}</span>`;
  });

  return { html: html.join("\n"), rootStyle };
}

/** Re-highlight every source block of one page. Returns the new HTML and the
 * number of blocks highlighted and skipped. */
export function highlightPage(highlighter, page) {
  let done = 0;
  const skipped = [];

  const out = page.replace(BLOCK, (whole, lang, inner) => {
    if (!LANGS.includes(lang)) {
      skipped.push(lang);
      return whole;
    }

    const result = highlightBlock(highlighter, lang, inner);

    if (result === null) {
      skipped.push(`${lang} (unexpected line markup)`);
      return whole;
    }

    done += 1;
    return `<code class="sourceCode ${lang}" data-shiki="" style="${result.rootStyle}">${result.html}</code>`;
  });

  return { html: out, done, skipped };
}

export async function makeHighlighter() {
  return createHighlighter({ themes: Object.values(THEMES), langs: LANGS });
}

function outputFiles() {
  const listFile = process.env.QUARTO_USE_FILE_FOR_PROJECT_OUTPUT_FILES;
  const list = listFile ? readFileSync(listFile, "utf8") : process.env.QUARTO_PROJECT_OUTPUT_FILES;

  if (list !== undefined) {
    return list.split("\n").filter((file) => file.endsWith(".html"));
  }

  return process.argv.slice(2);
}

async function main() {
  const highlighter = await makeHighlighter();
  let blocks = 0;
  const skipped = new Set();

  for (const file of outputFiles()) {
    const result = highlightPage(highlighter, readFileSync(file, "utf8"));
    writeFileSync(file, result.html);

    blocks += result.done;
    result.skipped.forEach((lang) => skipped.add(`${file}: ${lang}`));
  }

  console.log(`shiki: highlighted ${blocks} code blocks`);

  for (const entry of skipped) {
    console.warn(`shiki: left Quarto's highlighting on ${entry}`);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await main();
}
