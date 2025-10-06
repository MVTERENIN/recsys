// two-tower.js - TensorFlow.js implementation of Two-Tower retrieval model with MLPs

"use strict";

class TwoTowerModel {
  /*
    TwoTowerModel implements a dual-tower architecture for retrieval:
    - User Tower: embeddings for age, gender, occupation concatenated and passed through an MLP.
    - Item Tower: genre embeddings passed through an MLP.
    - Scoring: dot product of user and item embeddings.
    - Loss: supports in-batch sampled softmax (default) or BPR pairwise loss.
  */

  constructor(numUsers, numItems, embDim, hiddenUnits, numGenres, numOccupations, numGenders, lossType = 'softmax') {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim = embDim;
    this.hiddenUnits = hiddenUnits;
    this.numGenres = numGenres;
    this.numOccupations = numOccupations;
    this.numGenders = numGenders;
    this.lossType = lossType;

    // Embeddings for categorical features: gender, occupation
    this.genderEmbedding = tf.variable(tf.randomNormal([numGenders, embDim], 0, 0.05), true, 'genderEmbedding');
    this.occupationEmbedding = tf.variable(tf.randomNormal([numOccupations, embDim], 0, 0.05), true, 'occupationEmbedding');

    // Age embedding via dense projection instead of embedding table
    this.ageDenseW = tf.variable(tf.randomNormal([1, embDim], 0, 0.05), true, 'ageDenseW');
    this.ageDenseB = tf.variable(tf.zeros([embDim]), true, 'ageDenseB');

    // Item genres embedding projection matrix to embDim (learnable)
    this.genreEmbeddingW = tf.variable(tf.randomNormal([numGenres, embDim], 0, 0.05), true, 'genreEmbeddingW');

    // User MLP - input concatenated [ageEmb, genderEmb, occupationEmb] => hidden layer => final user embedding
    this.userMLP_W1 = tf.variable(tf.randomNormal([embDim * 3, hiddenUnits], 0, 0.1), true, 'userMLP_W1');
    this.userMLP_b1 = tf.variable(tf.zeros([hiddenUnits]), true, 'userMLP_b1');
    this.userMLP_W2 = tf.variable(tf.randomNormal([hiddenUnits, embDim], 0, 0.1), true, 'userMLP_W2');
    this.userMLP_b2 = tf.variable(tf.zeros([embDim]), true, 'userMLP_b2');

    // Item MLP - input embedded genres sum => hidden layer => final item embedding
    this.itemMLP_W1 = tf.variable(tf.randomNormal([embDim, hiddenUnits], 0, 0.1), true, 'itemMLP_W1');
    this.itemMLP_b1 = tf.variable(tf.zeros([hiddenUnits]), true, 'itemMLP_b1');
    this.itemMLP_W2 = tf.variable(tf.randomNormal([hiddenUnits, embDim], 0, 0.1), true, 'itemMLP_W2');
    this.itemMLP_b2 = tf.variable(tf.zeros([embDim]), true, 'itemMLP_b2');
  }

  // User forward pass: input tensor shape [batch, 3] with [ageScaled, genderIdx, occupationIdx]
  userForward(userFeatureTensor) {
    return tf.tidy(() => {
      // ageScaled is float; genderIdx and occupationIdx are integers
      const ageTensor = userFeatureTensor.slice([0, 0], [-1, 1]); // shape [batch,1]
      const genderIdxTensor = userFeatureTensor.slice([0, 1], [-1, 1]).toInt();
      const occIdxTensor = userFeatureTensor.slice([0, 2], [-1, 1]).toInt();

      // Age embedding by dense
      const ageEmb = ageTensor.matMul(this.ageDenseW).add(this.ageDenseB); // [batch, embDim]

      // Gender embedding lookup
      const genderEmb = tf.gather(this.genderEmbedding, genderIdxTensor.reshape([-1])); // [batch, embDim]

      // Occupation embedding lookup
      const occEmb = tf.gather(this.occupationEmbedding, occIdxTensor.reshape([-1])); // [batch, embDim]

      // Concatenate embeddings [batch, embDim*3]
      const concatEmb = tf.concat([ageEmb, genderEmb, occEmb], 1);

      // User MLP hidden layer with ReLU
      const h1 = concatEmb.matMul(this.userMLP_W1).add(this.userMLP_b1).relu();
      // Output user embedding
      const out = h1.matMul(this.userMLP_W2).add(this.userMLP_b2);
      return out; // shape [batch, embDim]
    });
  }

  // Item forward pass: input genre tensor shape [batch, numGenres] of 0/1
  itemForward(itemFeatureTensor) {
    return tf.tidy(() => {
      // Project genres via embedding matrix: [batch, numGenres] x [numGenres, embDim] => [batch, embDim]
      const genreEmb = itemFeatureTensor.matMul(this.genreEmbeddingW);

      // Item MLP hidden layer with ReLU
      const h1 = genreEmb.matMul(this.itemMLP_W1).add(this.itemMLP_b1).relu();
      // Output item embedding
      const out = h1.matMul(this.itemMLP_W2).add(this.itemMLP_b2);
      return out; // shape [batch, embDim]
    });
  }

  // Dot product scoring for pairs userEmb [batch, embDim], itemEmb [batch, embDim]
  score(userEmb, itemEmb) {
    return tf.tidy(() => {
      return tf.sum(userEmb.mul(itemEmb), 1); // shape [batch]
    });
  }

  // Training step with positive pairs (with in-batch negative sampling for softmax)
  trainStep(userFeatureTensor, posItemFeatureTensor, optimizer) {
    const lossType = this.lossType;

    const userMLPVars = [
      this.ageDenseW, this.ageDenseB,
      this.genderEmbedding, this.occupationEmbedding,
      this.userMLP_W1, this.userMLP_b1,
      this.userMLP_W2, this.userMLP_b2,
    ];
    const itemMLPVars = [
      this.genreEmbeddingW,
      this.itemMLP_W1, this.itemMLP_b1,
      this.itemMLP_W2, this.itemMLP_b2,
    ];

    const trainVars = userMLPVars.concat(itemMLPVars);

    const batchSize = userFeatureTensor.shape[0];

    const lossFn = () => {
      const userEmb = this.userForward(userFeatureTensor); // [batch, embDim]
      const posItemEmb = this.itemForward(posItemFeatureTensor); // [batch, embDim]

      if(lossType === 'softmax') {
        // Compute scores matrix logits: [batch, batch] = userEmb @ posItemEmb^T
        const logits = tf.matMul(userEmb, posItemEmb, false, true);
        // Labels are diagonal (positive pairs)
        const labels = tf.oneHot(tf.range(0, batchSize, 1, 'int32'), batchSize);
        // Softmax cross entropy loss
        const loss = tf.losses.softmaxCrossEntropy(labels, logits, undefined, undefined);
        return loss;
      } else if(lossType === 'bpr') {
        // BPR loss requires negatives - sample negatives in batch by shuffling posItemEmb
        const negItemEmb = tf.concat([posItemEmb.slice([1,0],[batchSize-1, -1]), posItemEmb.slice([0,0],[1, -1])], 0);
        const posScores = this.score(userEmb, posItemEmb);
        const negScores = this.score(userEmb, negItemEmb);
        const diff = posScores.sub(negScores);
        const loss = tf.neg(tf.logSigmoid(diff)).mean();
        return loss;
      }
      throw new Error(`Unknown loss type ${lossType}`);
    };

    const grads = tf.variableGrads(lossFn, trainVars);

    optimizer.applyGradients(grads.grads);
    grads.dispose();

    // Return scalar loss value for UI
    return lossFn();
  }
}

if(typeof module !== 'undefined') {
  module.exports = {TwoTowerModel};
}
