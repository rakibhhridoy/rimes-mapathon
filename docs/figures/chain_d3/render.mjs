/* Headless render of the chain diagram: node render.mjs
   Writes chain.svg; make_figures.py (or rsvg-convert) turns it into
   ../fig_chain.pdf. Counts come from counts.json. */
import * as d3 from "d3";
import { JSDOM } from "jsdom";
import fs from "fs";

const dom = new JSDOM("<!DOCTYPE html><body></body>", { pretendToBeVisual: true });
globalThis.document = dom.window.document;
eval(fs.readFileSync(new URL("./chain.js", import.meta.url), "utf8"));

const counts = JSON.parse(fs.readFileSync(new URL("./counts.json", import.meta.url), "utf8"));
const svg = d3.select(dom.window.document.body).append("svg")
  .attr("xmlns", "http://www.w3.org/2000/svg");
globalThis.drawChain(svg, d3, counts);

fs.writeFileSync(new URL("./chain.svg", import.meta.url),
  '<?xml version="1.0" encoding="UTF-8"?>\n' + dom.window.document.body.innerHTML);
console.log("wrote chain.svg");
