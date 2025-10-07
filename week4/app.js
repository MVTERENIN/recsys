// app.js - MovieLens 100K Two-Tower Demo
// Handles data loading, preprocessing, training loop, testing, UI updates, and visualization

"use strict";

const DATA_PATH = 'data/';
const MAX_INTERACTIONS_DEFAULT = 80000;
const EMBEDDING_DIM_DEFAULT = 32;
const HIDDEN_UNITS_DEFAULT = 64;
const LEARNING_RATE_DEFAULT = 0.001;
const EPOCHS_DEFAULT = 5;
const BATCH_SIZE_DEFAULT = 512;

// Global state containers
let interactions = []; // {userId, itemId, rating, ts}
let items = new Map(); // itemId -> {title, year, genres: Uint8Array(19)}
let users = new Map(); // userId -> {age, gender, occupation}

let userId2Idx = new Map();
let idx2UserId = new Map();
let itemId2Idx = new Map();
let idx2ItemId = new Map();

let userRatedItems = new Map(); // userIdx -> Set(itemIdx)
let userTopRatedItems = new Map(); // userIdx -> [{itemIdx, rating, ts} sorted by rating desc, ts desc]

let vocGender = new Map(); // map gender string to idx
let vocOccupation = new Map(); // map occupation string to idx

let model = null;
let optimizer = null;

let lossHistory = [];
let lossCanvas, lossCtx;
let embeddingCanvas, embeddingCtx;
let tooltip;

let trainingInProgress = false;

// Utility: parse genres string to binary array
const NUM_GENRES = 19;
const GENRE_NAMES = [
  "Unknown","Action","Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama",
  "Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"
];

// Load files from data folder and parse
async function loadData() {
  setStatus("Loading data files...");
  try {
    const [dataRaw, itemRaw, userRaw] = await Promise.all([
      fetch(DATA_PATH + "u.data").then(r => r.text()),
      fetch(DATA_PATH + "u.item").then(r => r.text()),
      fetch(DATA_PATH + "u.user").then(r => r.text()),
    ]);
    parseData(dataRaw, itemRaw, userRaw);
    setStatus(`Loaded data: ${interactions.length} interactions, ${items.size} items, ${users.size} users.`);
    document.getElementById('trainBtn').disabled = false;
    document.getElementById('testBtn').disabled = true;
  } catch (e) {
    setStatus(`Failed to load data: ${e.message}`);
  }
}

// Parse and build data structures
function parseData(uDataRaw, uItemRaw, uUserRaw) {
  interactions.length = 0;
  items.clear();
  users.clear();
  userId2Idx.clear();
  idx2UserId.clear();
  itemId2Idx.clear();
  idx2ItemId.clear();
  userRatedItems.clear();
  userTopRatedItems.clear();
  vocGender.clear();
  vocOccupation.clear();

  // Parse u.item - pipe separated, item_id|title|release_date|video_release_date|IMDb_URL|genres(19 binary)
  // Extract itemId, title, year (from title yyyy), genres (19 binaries)
  for (const line of uItemRaw.split('\n')) {
    if (!line.trim()) continue;
    const parts = line.split('|');
    if (parts.length < 24) continue;
    const itemId = parseInt(parts[0]);
    const titleRaw = parts[1];
    const yearMatch = titleRaw.match(/\((\d{4})\)$/);
    const year = yearMatch ? parseInt(yearMatch[1]) : 0;
    const title = yearMatch ? titleRaw.slice(0, yearMatch.index).trim() : titleRaw.trim();
    const genres = new Uint8Array(NUM_GENRES);
    for (let i = 0; i < NUM_GENRES; i++) {
      genres[i] = parts[5 + i] === '1' ? 1 : 0;
    }
    items.set(itemId, { title, year, genres });
  }

  // Parse u.user - user_id|age|gender|occupation|zip_code
  // Map genders and occupations to indices
  const genderSet = new Set();
  const occupationSet = new Set();
  for (const line of uUserRaw.split('\n')) {
    if (!line.trim()) continue;
    const parts = line.split('|');
    if (parts.length < 5) continue;
    const userId = parseInt(parts[0]);
    const age = parseInt(parts[1]);
    const gender = parts[2];
    const occupation = parts[3];
    genderSet.add(gender);
    occupationSet.add(occupation);
    users.set(userId, { age, gender, occupation });
  }
  // Build sorted vocabularies
  const sortedGenders = Array.from(genderSet).sort();
  for (let i = 0; i < sortedGenders.length; i++) vocGender.set(sortedGenders[i], i);
  const sortedOccupations = Array.from(occupationSet).sort();
  for (let i = 0; i < sortedOccupations.length; i++) vocOccupation.set(sortedOccupations[i], i);

  // Parse u.data - user_id, item_id, rating, timestamp (tab separated); limit to max interactions if needed
  for (const line of uDataRaw.split('\n')) {
    if (!line.trim()) continue;
    const parts = line.split('\t');
    if (parts.length < 4) continue;
    const userId = parseInt(parts[0]);
    const itemId = parseInt(parts[1]);
    const rating = parseInt(parts[2]);
    const ts = parseInt(parts[3]);
    if (!users.has(userId)) continue;
    if (!items.has(itemId)) continue;
    interactions.push({ userId, itemId, rating, ts });
  }

  // Re-index users and items 0-based
  const uniqueUserIds = Array.from(users.keys()).sort((a,b)=>a-b);
  uniqueUserIds.forEach((uid, i) => {
    userId2Idx.set(uid, i);
    idx2UserId.set(i, uid);
  });
  const uniqueItemIds = Array.from(items.keys()).sort((a,b)=>a-b);
  uniqueItemIds.forEach((iid, i) => {
    itemId2Idx.set(iid, i);
    idx2ItemId.set(i, iid);
  });

  // Build userRatedItems and userTopRatedItems maps with 0-based indices
  userRatedItems.clear();
  userTopRatedItems.clear();

  const tempUserRatings = new Map();
  for (const { userId, itemId, rating, ts } of interactions) {
    const uIdx = userId2Idx.get(userId);
    const iIdx = itemId2Idx.get(itemId);
    if (!userRatedItems.has(uIdx)) userRatedItems.set(uIdx, new Set());
    userRatedItems.get(uIdx).add(iIdx);
    if(!tempUserRatings.has(uIdx)) tempUserRatings.set(uIdx, []);
    tempUserRatings.get(uIdx).push({ itemIdx: iIdx, rating, ts });
  }
  // Sort userTopRatedItems: rating desc, ts desc and save top lists
  for (const [uIdx, ratedList] of tempUserRatings.entries()) {
    ratedList.sort((a,b) => {
      if(b.rating !== a.rating) return b.rating - a.rating;
      return b.ts - a.ts;
    });
    userTopRatedItems.set(uIdx, ratedList);
  }
}

// Content-based genre cosine similarity recommendation for a user, exclude rated items
function contentBasedRecommend(userIdx, topN=10) {
  // Average genres of user's rated items weighted by ratings
  const rated = userTopRatedItems.get(userIdx);
  if(!rated || rated.length === 0) return [];

  const userGenreSum = new Float32Array(NUM_GENRES).fill(0);
  let totalWeight = 0;
  for (const { itemIdx, rating } of rated) {
    const item = items.get(idx2ItemId.get(itemIdx));
    if(!item) continue;
    for(let g=0; g<NUM_GENRES; g++) {
      userGenreSum[g] += item.genres[g] * rating;
    }
    totalWeight += rating;
  }
  if(totalWeight === 0) return [];
  for(let g=0; g<NUM_GENRES; g++) userGenreSum[g] /= totalWeight;

  // Cosine similarity function
  function cosineSim(vecA, vecB) {
    let dot=0, magA=0, magB=0;
    for(let i=0; i<NUM_GENRES; i++) {
      dot += vecA[i]*vecB[i];
      magA += vecA[i]*vecA[i];
      magB += vecB[i]*vecB[i];
    }
    return magA===0||magB===0 ? 0 : dot/(Math.sqrt(magA)*Math.sqrt(magB));
  }

  const ratedSet = userRatedItems.get(userIdx) || new Set();

  // Score all items by similarity excluding rated
  let scoredItems = [];
  for (const [itemId, item] of items.entries()) {
    const iIdx = itemId2Idx.get(itemId);
    if(ratedSet.has(iIdx)) continue;
    const score = cosineSim(userGenreSum, item.genres);
    scoredItems.push({ iIdx, score });
  }
  scoredItems.sort((a,b) => b.score - a.score);
  return scoredItems.slice(0, topN).map(({iIdx}) => idx2ItemId.get(iIdx));
}

// Prepare user features tensor for model input from userIdx
function prepareUserFeaturesTensor(userIdx) {
  const userId = idx2UserId.get(userIdx);
  const user = users.get(userId);
  const ageNorm = Math.min(Math.max(user.age, 5), 90); // clamp age
  // Normalize age roughly to [0,1]
  const ageScaled = (ageNorm - 5) / (90 - 5);
  const genderIdx = vocGender.get(user.gender) || 0;
  const occupationIdx = vocOccupation.get(user.occupation) || 0;

  // Tensor shape [1,3], columns: ageScaled (float), genderIdx (int), occupationIdx (int)
  return tf.tensor2d([[ageScaled, genderIdx, occupationIdx]], [1,3]);
}

// Prepare item features tensor from item index (genres)
function prepareItemFeaturesTensor(itemIdx) {
  const itemId = idx2ItemId.get(itemIdx);
  const item = items.get(itemId);
  // genres: Uint8Array NUM_GENRES length
  // Shape [1,NUM_GENRES]
  return tf.tensor2d([Array.from(item.genres)], [1, NUM_GENRES]);
}

// Train loop state and config
let trainingConfig = {
  epochs: EPOCHS_DEFAULT,
  batchSize: BATCH_SIZE_DEFAULT,
  embeddingDim: EMBEDDING_DIM_DEFAULT,
  learningRate: LEARNING_RATE_DEFAULT,
  maxInteractions: MAX_INTERACTIONS_DEFAULT,
  hiddenUnits: HIDDEN_UNITS_DEFAULT,
  lossType: 'softmax', // or 'bpr'
}

// Build training batches with in-batch negatives
function* batchGenerator(shuffledInteractions, batchSize) {
  for(let i=0; i<shuffledInteractions.length; i+=batchSize) {
    yield shuffledInteractions.slice(i, i+batchSize);
  }
}

// Shuffle array inplace (Fisher-Yates)
function shuffleArray(arr) {
  for(let i=arr.length-1; i>0; i--) {
    const j = Math.floor(Math.random()*(i+1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

// Train function async with UI updates
async function trainModel() {
  if(trainingInProgress) return;
  trainingInProgress = true;
  setStatus("Training started...");
  lossHistory.length = 0;
  clearLossChart();
  clearEmbeddingCanvas();
  document.getElementById('testBtn').disabled = true;

  // Initialize model if not created
  if(!model) {
    model = new TwoTowerModel(
      userId2Idx.size,
      itemId2Idx.size,
      trainingConfig.embeddingDim,
      trainingConfig.hiddenUnits,
      NUM_GENRES,
      vocOccupation.size,
      vocGender.size,
      trainingConfig.lossType
    );
    optimizer = tf.train.adam(trainingConfig.learningRate);
  }

  // Use subset of interactions if larger than maxInteractions
  let trainInteractions = interactions;
  if(interactions.length > trainingConfig.maxInteractions) {
    trainInteractions = interactions.slice(0, trainingConfig.maxInteractions);
  }

  // Shuffle interactions
  const shuffled = [...trainInteractions];
  shuffleArray(shuffled);

  // Convert interaction entries to indices for training
  const trainEntries = shuffled.map(({userId, itemId}) => {
    return {
      uIdx: userId2Idx.get(userId),
      iIdx: itemId2Idx.get(itemId)
    };
  });

  const numBatches = Math.ceil(trainEntries.length / trainingConfig.batchSize);

  for(let epoch=0; epoch<trainingConfig.epochs; epoch++) {
    let epochLossSum = 0;
    let epochCount = 0;

    let batchNum = 0;
    for (const batch of batchGenerator(trainEntries, trainingConfig.batchSize)) {
      batchNum++;
      const lossVal = await tf.tidy(() => {
        // Prepare tensors for batch
        const batchSizeActual = batch.length;

        const userAges = [];
        const userGenders = [];
        const userOccupations = [];
        const posItemGenres = [];

        for(const {uIdx, iIdx} of batch) {
          const userId = idx2UserId.get(uIdx);
          const user = users.get(userId);
          // Normalize age in [0,1]
          const ageNorm = Math.min(Math.max(user.age,5),90);
          userAges.push((ageNorm -5)/(90-5));
          userGenders.push(vocGender.get(user.gender)||0);
          userOccupations.push(vocOccupation.get(user.occupation)||0);
          const itemId = idx2ItemId.get(iIdx);
          const item = items.get(itemId);
          posItemGenres.push(Array.from(item.genres));
        }

        // Construct user and item tensors
        const userFeatureTensor = tf.tensor2d(userAges.map((v,i) => [v, userGenders[i], userOccupations[i]]));
        const posItemFeaturesTensor = tf.tensor2d(posItemGenres);

        // Minibatch negatives from in-batch items
        // item embeddings matrix for batch items
        const negItemFeaturesTensor = posItemFeaturesTensor;

        // Compute loss and gradients via gradient tape
        // --- FIX: Proper disposal of gradients ---
        const grads = tf.variableGrads(() => model.trainStep(userFeatureTensor, posItemFeaturesTensor, optimizer), model.trainVars);
        optimizer.applyGradients(grads.grads);
        for (let key in grads.grads) {
          grads.grads[key].dispose();
        }
        // --- End fix ---
        // Return scalar loss value for UI
        const lossVal = grads.value.dataSync()[0];

        userFeatureTensor.dispose();
        posItemFeaturesTensor.dispose();
        negItemFeaturesTensor.dispose();

        return lossVal;
      });

      epochLossSum += lossVal;
      epochCount++;

      // Update loss chart live
      const avgLoss = epochLossSum / epochCount;
      lossHistory.push(avgLoss);
      updateLossChart(lossHistory, epoch+1, numBatches, batchNum);

      await tf.nextFrame(); // yield to UI
    }
    setStatus(`Epoch ${epoch+1}/${trainingConfig.epochs} completed, avg loss: ${(epochLossSum/epochCount).toFixed(4)}`);
  }
  trainingInProgress = false;
  setStatus("Training completed. Visualizing embeddings...");
  await drawEmbeddingProjection();
  document.getElementById('testBtn').disabled = false;
}

// Draw loss chart on canvas
function updateLossChart(lossArr, epoch, totalBatches, currentBatch) {
  if(!lossCtx) return;
  clearLossChart();
  const width = lossCanvas.width;
  const height = lossCanvas.height;
  lossCtx.strokeStyle = "#4a90e2";
  lossCtx.lineWidth = 2;
  lossCtx.beginPath();

  // Draw loss curve: x = lossArr length, y = loss value scaled
  const maxLoss = Math.max(...lossArr);
  const minLoss = Math.min(...lossArr);
  const rangeLoss = maxLoss - minLoss || 1;
  const length = lossArr.length;

  // Plot points scaled horizontally and vertically
  for(let i=0; i<length; i++) {
    const x = (i / (length - 1)) * width;
    const y = height - ((lossArr[i] - minLoss) / rangeLoss) * height;
    if(i===0) lossCtx.moveTo(x, y);
    else lossCtx.lineTo(x, y);
  }
  lossCtx.stroke();

  // Draw axes and text
  lossCtx.fillStyle = "#222";
  lossCtx.font = "12px Arial";
  lossCtx.fillText(`Epoch: ${epoch} Batch: ${currentBatch}/${totalBatches}`, 10, 20);
  lossCtx.fillText(`Loss range: ${minLoss.toFixed(3)} - ${maxLoss.toFixed(3)}`, 10, 40);
}

function clearLossChart() {
  if(!lossCtx) return;
  lossCtx.clearRect(0,0,lossCanvas.width, lossCanvas.height);
}

function clearEmbeddingCanvas() {
  if(!embeddingCtx) return;
  embeddingCtx.clearRect(0,0,embeddingCanvas.width,embeddingCanvas.height);
  tooltip.style.visibility = "hidden";
}

// Draw PCA 2D projection of a sample of item embeddings with hover showing titles
async function drawEmbeddingProjection() {
  if(!model) return;
  // Sample up to 1000 items randomly or all if fewer
  const totalItems = itemId2Idx.size;
  const SAMPLE_COUNT = Math.min(1000, totalItems);
  const allItemIndices = [...Array(totalItems).keys()];
  shuffleArray(allItemIndices);
  const sampleIndices = allItemIndices.slice(0, SAMPLE_COUNT);

  // Get genre features for sample items
  const sampleGenres = sampleIndices.map(iIdx => {
    const itemId = idx2ItemId.get(iIdx);
    return Array.from(items.get(itemId).genres);
  });

  // Compute item embeddings for sample items
  const itemFeaturesTensor = tf.tensor2d(sampleGenres);

  // Forward pass item tower to get final embeddings shape [SAMPLE_COUNT, embDim]
  const itemEmbeddings = model.itemForward(itemFeaturesTensor);

  // Run PCA on embeddings: center data and do power iteration on covariance to get top 2 components
  const embeddings2d = await pca2d(itemEmbeddings);

  // Draw scatter plot on embeddingCanvas
  clearEmbeddingCanvas();
  const ctx = embeddingCtx;
  const width = embeddingCanvas.width;
  const height = embeddingCanvas.height;

  // Normalize embeddings2d to canvas coords
  const xs = embeddings2d.map(p => p[0]);
  const ys = embeddings2d.map(p => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  function normX(x) { return 40 + ((x - minX) / (maxX-minX)) * (width-80); }
  function normY(y) { return height - (40 + ((y - minY) / (maxY-minY)) * (height-80)); }

  // Draw points
  ctx.fillStyle = "#4285f4";
  ctx.strokeStyle = "#1a1a1a";
  ctx.lineWidth = 0.5;
  ctx.clearRect(0,0,width,height);

  // Store circle positions for tooltip
  const circles = [];
  for(let i=0; i<embeddings2d.length; i++) {
    const cx = normX(xs[i]);
    const cy = normY(ys[i]);
    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, 2*Math.PI);
    ctx.fill();
    ctx.stroke();
    circles.push({cx, cy, itemIdx: sampleIndices[i]});
  }

  // Handle mousemove to show tooltip title on hover
  embeddingCanvas.onmousemove = e => {
    const rect = embeddingCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const radius = 5;
    let found = false;
    for(const c of circles) {
      const dist = Math.sqrt((c.cx - mx)**2 + (c.cy - my)**2);
      if(dist < radius) {
        const itemId = idx2ItemId.get(c.itemIdx);
        const title = items.get(itemId).title;
        tooltip.style.left = `${e.clientX+10}px`;
        tooltip.style.top = `${e.clientY+10}px`;
        tooltip.textContent = title;
        tooltip.style.visibility = "visible";
        found = true;
        break;
      }
    }
    if(!found) tooltip.style.visibility = "hidden";
  };

  itemFeaturesTensor.dispose();
  itemEmbeddings.dispose();

  setStatus("Embedding projection completed.");
}

// PCA with numeric approximation (power method) for 2 principal components
async function pca2d(embeddingsTensor) {
  // embeddingsTensor shape: [N, D]
  // Center data
  const mean = embeddingsTensor.mean(0);
  const centered = embeddingsTensor.sub(mean);

  // Compute covariance matrix covariance = centered^T * centered / N
  const cov = tf.matMul(centered.transpose(), centered).div(embeddingsTensor.shape[0]);

  // Use power iteration to find 1st eigenvector
  async function powerIteration(matrix, numIter=50) {
    let b = tf.randomNormal([matrix.shape[0],1]);
    for(let i=0; i<numIter; i++) {
      let b1 = tf.matMul(matrix, b);
      const norm = b1.norm();
      b = b1.div(norm);
      await tf.nextFrame();
    }
    return b;
  }

  // Find top eigenvector and eigenvalue
  const v1 = await powerIteration(cov);
  const lambda1 = tf.matMul(v1.transpose(), tf.matMul(cov, v1)).dataSync()[0];

  // Deflate covariance matrix
  const outer = tf.matMul(v1, v1.transpose()).mul(lambda1);
  const cov2 = cov.sub(outer);

  // Find second eigenvector v2
  const v2 = await powerIteration(cov2);

  // Project centered data to components
  const comp1 = tf.matMul(centered, v1).reshape([-1]);
  const comp2 = tf.matMul(centered, v2).reshape([-1]);

  const comp1Arr = await comp1.array();
  const comp2Arr = await comp2.array();

  mean.dispose();
  centered.dispose();
  cov.dispose();
  v1.dispose();
  v2.dispose();
  comp1.dispose();
  comp2.dispose();
  outer.dispose();
  cov2.dispose();

  const projected = [];
  for(let i=0; i<comp1Arr.length; i++) {
    projected.push([comp1Arr[i], comp2Arr[i]]);
  }
  return projected;
}

// Pick random user with >=20 ratings
function pickRandomQualifiedUser(minRatings=20) {
  const candidates = [];
  for(const [uIdx, rated] of userTopRatedItems.entries()) {
    if(rated.length >= minRatings) candidates.push(uIdx);
  }
  if(candidates.length === 0) return null;
  const r = Math.floor(Math.random()*candidates.length);
  return candidates[r];
}

// Test function: for random qualified user display side-by-side tables
async function testRecommendations() {
  setStatus("Testing recommendations...");
  if(!model) {
    setStatus("Train model before testing.");
    return;
  }
  const userIdx = pickRandomQualifiedUser(20);
  if(userIdx === null) {
    setStatus("No user with >=20 ratings found.");
    return;
  }
  // Get user's historically top-rated 10 movies by rating desc then recency desc
  const userRated = userTopRatedItems.get(userIdx).slice(0, 10).map(({itemIdx}) => idx2ItemId.get(itemIdx));

  // Deep Learning model recommendations: compute user embedding, score all items, exclude rated, top-10
  const userFeatureTensor = prepareUserFeaturesTensor(userIdx);
  const userEmb = model.userForward(userFeatureTensor);

  // Compute item embeddings matrix in batches for memory control
  const BATCH_SIZE = 256;
  const totalItems = itemId2Idx.size;
  let scores = new Float32Array(totalItems);

  for(let start=0; start<totalItems; start+=BATCH_SIZE) {
    const end = Math.min(start+BATCH_SIZE, totalItems);
    const batchGenres = [];
    for(let i=start; i<end; i++) {
      const itemId = idx2ItemId.get(i);
      batchGenres.push(Array.from(items.get(itemId).genres));
    }
    const itemTensor = tf.tensor2d(batchGenres);
    const itemEmb = model.itemForward(itemTensor);
    const userEmbBatch = userEmb.reshape([1, userEmb.shape[1]]);
    const batchScores = tf.matMul(userEmbBatch, itemEmb.transpose()).reshape([-1]);
    const batchScoresArr = await batchScores.array();
    for(let i=0; i<batchScoresArr.length; i++) {
      scores[start+i] = batchScoresArr[i];
    }
    itemTensor.dispose();
    itemEmb.dispose();
    batchScores.dispose();
  }
  userEmb.dispose();
  userFeatureTensor.dispose();

  const ratedSet = userRatedItems.get(userIdx);
  // Exclude rated items
  const scoredItemsFiltered = [];
  for(let i=0; i<totalItems; i++) {
    if(ratedSet.has(i)) continue;
    scoredItemsFiltered.push({iIdx: i, score: scores[i]});
  }
  scoredItemsFiltered.sort((a,b)=>b.score - a.score);
  const dlTop10 = scoredItemsFiltered.slice(0,10).map(({iIdx}) => idx2ItemId.get(iIdx));

  // Content-based recommendations
  const cbTop10 = contentBasedRecommend(userIdx, 10);

  // Render side-by-side HTML table with 3 columns
  renderResultsTable(userIdx, userRated, dlTop10, cbTop10);

  setStatus("Test completed. Recommendations displayed.");
}

// Render results table with three columns: historic top-10, deep learning top-10, content-based top-10
function renderResultsTable(userIdx, histList, dlRecoList, cbRecoList) {
  const container = document.getElementById('resultsContainer');
  container.innerHTML = "";

  const table = document.createElement('table');
  table.id = "sideBySideTable";
  const thead = document.createElement('thead');
  const trhead = document.createElement('tr');
  ["Top-10 Historically Rated", "Top-10 DL Recommended", "Top-10 Content-Based"].forEach(title => {
    const th = document.createElement('th');
    th.textContent = title;
    trhead.appendChild(th);
  });
  thead.appendChild(trhead);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');

  // max rows = 10 fixed
  for(let i=0; i<10; i++) {
    const tr = document.createElement('tr');

    // Historic top-10
    const tdHist = document.createElement('td');
    if (i < histList.length) {
      const item = items.get(histList[i]);
      tdHist.textContent = `${item.title} (${item.year})`;
    }
    tr.appendChild(tdHist);

    // DL top-10
    const tdDL = document.createElement('td');
    if (i < dlRecoList.length) {
      const item = items.get(dlRecoList[i]);
      tdDL.textContent = `${item.title} (${item.year})`;
    }
    tr.appendChild(tdDL);

    // Content-based top-10
    const tdCB = document.createElement('td');
    if (i < cbRecoList.length) {
      const item = items.get(cbRecoList[i]);
      tdCB.textContent = `${item.title} (${item.year})`;
    }
    tr.appendChild(tdCB);

    tbody.appendChild(tr);
  }

  table.appendChild(tbody);
  container.appendChild(table);
}

// Set status text area
function setStatus(txt) {
  const statusDiv = document.getElementById('status');
  statusDiv.textContent = txt;
}

// Initialize canvas contexts and event listeners on DOM ready
function init() {
  lossCanvas = document.getElementById('lossChart');
  lossCtx = lossCanvas.getContext('2d');
  embeddingCanvas = document.getElementById('embeddingCanvas');
  embeddingCtx = embeddingCanvas.getContext('2d');
  tooltip = document.getElementById('tooltip');

  document.getElementById('loadDataBtn').onclick = async () => {
    await loadData();
  };
  document.getElementById('trainBtn').onclick = async () => {
    await trainModel();
  };
  document.getElementById('testBtn').onclick = async () => {
    await testRecommendations();
  };

  setStatus("Ready. Click 'Load Data' to start.");
}

window.onload = init;

