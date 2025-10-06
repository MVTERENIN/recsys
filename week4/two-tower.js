class TwoTowerModel {
    constructor(numUsers, numItems, numGenres, config) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.numGenres = numGenres;
        this.config = config;
        
        this.buildModel();
        this.initializeOptimizer();
        
        console.log('TwoTowerModel initialized:', {
            numUsers,
            numItems, 
            numGenres,
            config
        });
    }

    buildModel() {
        const { embeddingDim } = this.config;
        
        // User tower
        this.userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: embeddingDim,
            name: 'user_embedding'
        });
        
        this.userMLP = tf.sequential({
            layers: [
                tf.layers.dense({units: 32, activation: 'relu', name: 'user_dense1'}),
                tf.layers.dense({units: embeddingDim, activation: 'linear', name: 'user_output'})
            ]
        });
        
        // Item tower  
        this.itemEmbedding = tf.layers.embedding({
            inputDim: this.numItems,
            outputDim: embeddingDim, 
            name: 'item_embedding'
        });
        
        this.itemMLP = tf.sequential({
            layers: [
                tf.layers.dense({units: 32, activation: 'relu', name: 'item_dense1'}),
                tf.layers.dense({units: embeddingDim, activation: 'linear', name: 'item_output'})
            ]
        });
        
        console.log('Model architecture built');
    }

    initializeOptimizer() {
        this.optimizer = tf.train.adam(this.config.learningRate);
    }

    userForward(userIds) {
        return tf.tidy(() => {
            const userEmb = this.userEmbedding.apply(userIds);
            const squeezed = tf.squeeze(userEmb, [1]);
            const userMLPOut = this.userMLP.apply(squeezed);
            return tf.l2Normalize(userMLPOut, -1);
        });
    }

    itemForward(itemIds) {
        return tf.tidy(() => {
            const itemEmb = this.itemEmbedding.apply(itemIds);
            const squeezed = tf.squeeze(itemEmb, [1]);
            const itemMLPOut = this.itemMLP.apply(squeezed);
            return tf.l2Normalize(itemMLPOut, -1);
        });
    }

    computeLoss(userEmbeddings, itemEmbeddings) {
        return tf.tidy(() => {
            const batchSize = userEmbeddings.shape[0];
            const scores = tf.matMul(userEmbeddings, itemEmbeddings, false, true);
            const labels = tf.oneHot(tf.range(0, batchSize), batchSize);
            return tf.losses.softmaxCrossEntropy(labels, scores);
        });
    }

    async trainBatch(batchInteractions, userIdMap, itemIdMap) {
        return tf.tidy(() => {
            const batchSize = batchInteractions.length;
            
            const userIds = batchInteractions.map(i => userIdMap.get(i.userId));
            const itemIds = batchInteractions.map(i => itemIdMap.get(i.itemId));
            
            const userIdsTensor = tf.tensor2d(userIds, [batchSize, 1], 'int32');
            const itemIdsTensor = tf.tensor2d(itemIds, [batchSize, 1], 'int32');
            
            const loss = this.optimizer.minimize(() => {
                const userEmbs = this.userForward(userIdsTensor);
                const itemEmbs = this.itemForward(itemIdsTensor);
                return this.computeLoss(userEmbs, itemEmbs);
            }, true);
            
            return loss ? loss.dataSync()[0] : 0;
        });
    }

    async trainEpoch(interactions, userIdMap, itemIdMap) {
        const { batchSize, maxInteractions } = this.config;
        
        // Use random sample for training
        const shuffled = [...interactions]
            .sort(() => Math.random() - 0.5)
            .slice(0, Math.min(maxInteractions, interactions.length));
        
        const numBatches = Math.ceil(shuffled.length / batchSize);
        let totalLoss = 0;
        let processedBatches = 0;
        
        for (let batch = 0; batch < numBatches; batch++) {
            const start = batch * batchSize;
            const end = Math.min(start + batchSize, shuffled.length);
            const batchData = shuffled.slice(start, end);
            
            if (batchData.length === 0) continue;
            
            const loss = await this.trainBatch(batchData, userIdMap, itemIdMap);
            totalLoss += loss;
            processedBatches++;
            
            // Clean up memory
            tf.engine().startScope();
            tf.engine().endScope();
            
            // Update progress occasionally
            if (batch % 5 === 0) {
                console.log(`Batch ${batch}/${numBatches}, loss: ${loss.toFixed(4)}`);
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }
        
        return processedBatches > 0 ? totalLoss / processedBatches : 1.0;
    }

    async getUserEmbedding(userIdx) {
        return tf.tidy(() => {
            const userIdTensor = tf.tensor2d([[userIdx]], [1, 1], 'int32');
            return this.userForward(userIdTensor);
        });
    }

    async getItemEmbedding(itemIdx) {
        return tf.tidy(() => {
            const itemIdTensor = tf.tensor2d([[itemIdx]], [1, 1], 'int32');
            return this.itemForward(itemIdTensor);
        });
    }

    async scoreUserItem(userEmbedding, itemIdx) {
        return tf.tidy(async () => {
            const itemEmbedding = await this.getItemEmbedding(itemIdx);
            const score = tf.sum(tf.mul(userEmbedding, itemEmbedding));
            const scoreValue = score.dataSync()[0];
            
            // Cleanup
            itemEmbedding.dispose();
            score.dispose();
            
            return scoreValue;
        });
    }
}
