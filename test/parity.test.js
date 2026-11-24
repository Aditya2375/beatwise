'use strict';
const test = require('node:test'), assert = require('node:assert');
const fs = require('fs'), path = require('path');
const BM = require('../js/model.js');
const root = path.join(__dirname, '..');
const W = JSON.parse(fs.readFileSync(path.join(root, 'data/weights.json')));
const recs = JSON.parse(fs.readFileSync(path.join(root, 'data/records.json')));
const fx = JSON.parse(fs.readFileSync(path.join(__dirname, 'parity.json')));
test('JS features and probabilities match the Python implementation on real beats', () => {
  let worstF = 0, worstP = 0, n = 0;
  for (const item of fx) {
    const rec = recs.find((r) => r.id === item.id);
    item.ks.forEach((k, j) => {
      if (j % 2) return;
      const f = BM.features(rec, k), p = BM.predictFeatures(W, f);
      f.forEach((v, i) => { worstF = Math.max(worstF, Math.abs(v - item.f[j][i])); });
      p.forEach((v, i) => { worstP = Math.max(worstP, Math.abs(v - item.p[j][i])); });
      n++;
    });
  }
  assert.ok(n > 100, 'enough beats checked');
  assert.ok(worstF < 2e-3, 'feature drift ' + worstF);
  assert.ok(worstP < 5e-3, 'probability drift ' + worstP);
});
