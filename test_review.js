const test = require('node:test');
const assert = require('node:assert/strict');
const {labelForReview, reviewedCsv, labeledText} = require('./ui/review.js');

test('correct and incorrect map binary decisions to reviewer labels', () => {
  assert.equal(labelForReview('looks_gibberish', 'correct'), 'flagged');
  assert.equal(labelForReview('looks_gibberish', 'incorrect'), 'ordinary');
  assert.equal(labelForReview('looks_ordinary', 'correct'), 'ordinary');
  assert.equal(labelForReview('looks_ordinary', 'incorrect'), 'flagged');
  assert.equal(labelForReview('unsure', 'incorrect'), null);
  assert.equal(labelForReview('unsure', 'flagged'), 'flagged');
  assert.equal(labelForReview('score_only', 'uncertain'), null);
});

test('exports reviewed decisions while retaining uncertainty and excluding unlabeled training rows', () => {
  const rows = [
    {line:1,email:'mukunda@example.com',decision:'looks_ordinary',review:'correct',model:'demo'},
    {line:2,email:'qzx@example.com',decision:'looks_gibberish',review:'incorrect',model:'demo'},
    {line:3,email:'bad@example.com',decision:'unsure',review:'uncertain',model:'demo'},
    {line:4,email:'ignore@example.com',decision:'looks_gibberish',model:'demo'},
    {line:5,email:'=unsafe@example.com',decision:'unsure',review:'flagged',model:'demo'},
    {line:6,email:'line\nbreak@example.com',decision:'unsure',review:'flagged',model:'demo'}
  ];
  const csv = reviewedCsv(rows);
  assert.match(csv, /"uncertain",""/);
  assert.match(csv, /"'=unsafe@example.com"/);
  assert.doesNotMatch(csv, /ignore@example.com/);
  assert.equal(labeledText(rows, 'ordinary'), 'mukunda@example.com\nqzx@example.com\n');
  assert.equal(labeledText(rows, 'flagged'), '=unsafe@example.com\n');
});
