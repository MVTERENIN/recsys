class TwoTowerModel {
    constructor(numUsers, numItems, embDim, hiddenUnits, numGenres, numOccupations, numGenders, learningRate) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embDim = embDim;
        this.hiddenUnits = hiddenUnits;
        this.numGenres = numGenres;
        this.numOccupations = numOccupations;
        this.numGenders = numGenders;
        
        this.optimizer = tf.train.adam(learningRate);
        
        // Build occupation and gender mappings
        this.occupationList = null;
        this.genderMap = { 'M': 0, 'F': 1 };
        
        this.buildModel();
    }

    buildModel() {
        // User tower components
        this.userIdEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: this.embDim,
            name: 'user_id_embedding'
        });

        this.ageNormalization = tf.layers.dense({
            units: 1,
            activation: 'linear',
            useBias: true,
            name: 'age_normalization'
        });

        this.genderEmbedding = tf.layers.embedding({
            inputDim: this.numGenders,
            outputDim: Math.max(2, Math.floor(this.embDim / 4)),
            name: 'gender_embedding'
        });

        this.occupationEmbedding = tf.layers.embedding({
            inputDim: this.numOccupations,
            outputDim: Math.max(4, Math.floor(this.embDim / 2)),
            name: 'occupation_embedding'
        });

        // User MLP layers
        this.userLayers = [];
        for (const units of this.hiddenUnits) {
            this.userLayers.push(
                tf.layers.dense({
                    units: units,
                    activation: 'relu',
                    name: `user_dense_${units}`
                })
            );
        }
        this.userOutputLayer = tf.layers.dense({
            units: this.embDim,
            activation: 'linear',
            name: 'user_output'
        });

        // Item tower
        this.itemProjection = tf.layers.dense({
            units: this.embDim,
            activation: 'linear',
            name: 'item_projection'
        });

        // Item MLP layers
        this.itemLayers = [];
        for (const units of this.hiddenUnits) {
            this.itemLayers.push(
                tf.layers.dense({
                    units: units,
                    activation: 'relu',
                    name: `item_dense_${units}`
                })
            );
        }
        this.itemOutputLayer = tf.layers.dense({
            units: this.embDim,
            activation: 'linear',
            name: 'item_output'
        });
    }

    userTower(userIds, userFeaturesList, app) {
        return tf.tidy(() => {
            const batchSize = userIds.length;
            
            // User ID embeddings
            const userIdTensor = tf.tensor1d(userIds, 'int32');
            const userEmbs = this.userIdEmbedding.apply(userIdTensor);
            
            // Age normalization
            const ages = userFeaturesList.map(f => f.age / 100.0);
            const ageTensor = tf.tensor1d(ages, 'float32').expandDims(1);
            const ageNorm = this.ageNormalization.apply(ageTensor);
            
            // Gender embeddings
            const genderIndices = userFeaturesList.map(f => this.genderMap[f.gender] || 0);
            const genderTensor = tf.tensor1d(genderIndices, 'int32');
            const genderEmbs = this.genderEmbedding.apply(genderTensor);
            
            // Occupation embeddings
            if (!this.occupationList) {
                this.occupationList = [...new Set([...app.users.values()].map(u => u.occupation))];
            }
            const occupationIndices = userFeaturesList.map(f => {
                const index = this.occupationList.indexOf(f.occupation);
                return Math.max(0, index);
            });
            const occupationTensor = tf.tensor1d(occupationIndices, 'int32');
            const occupationEmbs = this.occupationEmbedding.apply(occupationTensor);
            
            // Concatenate all features
            let combined = tf.concat([
                userEmbs,
                ageNorm,
                genderEmbs,
                occupationEmbs
            ], 1);
            
            // Apply MLP
            for (const layer of this.userLayers) {
                combined = layer.apply(combined);
            }
            
            return this.userOutputLayer.apply(combined);
        });
    }

    itemTower(itemFeaturesBatch) {
        return tf.tidy(() => {
            let features = tf.tensor2d(itemFeaturesBatch);
            features = this.itemProjection.apply(features);
            
            // Apply MLP
            for (const layer of this.itemLayers) {
                features = layer.apply(features);
            }
            
            return this.itemOutputLayer.apply(features);
        });
    }

    async trainBatch(batchPairs, app) {
        // Extract batch data
        const userIds = batchPairs.map(p => p.userIndex);
        const itemIds = batchPairs.map(p => p.itemIndex);
        const userFeatures = batchPairs.map(p => p.userFeatures);
        
        // Get item features
        const itemFeatures = itemIds.map(itemId => {
            const itemData = app.items.get(app.indexToItemId.get(itemId));
            return itemData.genres;
        });

        const loss = await this.trainStep(userIds, userFeatures, itemFeatures, app);
        return loss;
    }

    async trainStep(userIds, userFeatures, itemFeatures, app) {
        return tf.tidy(() => {
            // Forward pass - user tower (batch processing)
            const userEmbeddings = this.userTower(userIds, userFeatures, app);
            
            // Forward pass - item tower (batch processing)
            const itemEmbeddings = this.itemTower(itemFeatures);
            
            // Compute similarity matrix (batch_size x batch_size)
            const scores = tf.matMul(userEmbeddings, itemEmbeddings, false, true);
            
            // In-batch negative sampling: diagonal are positives
            const batchSize = userIds.length;
            const labels = tf.oneHot(tf.range(0, batchSize, 1, 'int32'), batchSize);
            
            // Softmax cross entropy loss
            const loss = tf.losses.softmaxCrossEntropy(labels, scores).mean();
            
            // Compute gradients and update
            const variables = this.getTrainableVariables();
            const gradients = tf.grad(l => l).call(this, loss, variables);
            
            this.optimizer.applyGradients(
                variables.map((v, i) => ({ grad: gradients[i], var: v }))
            );
            
            return loss.dataSync()[0];
        });
    }

    getTrainableVariables() {
        const layers = [
            this.userIdEmbedding, this.ageNormalization, this.genderEmbedding,
            this.occupationEmbedding, ...this.userLayers, this.userOutputLayer,
            this.itemProjection, ...this.itemLayers, this.itemOutputLayer
        ];
        
        return layers.flatMap(layer => layer.trainableWeights);
    }

    async getUserEmbedding(userIndex, userFeatures, app) {
        return tf.tidy(() => {
            const userIds = [userIndex];
            const userFeaturesList = [userFeatures];
            const embedding = this.userTower(userIds, userFeaturesList, app);
            const data = embedding.dataSync();
            return Array.from(data);
        });
    }

    async scoreUserItems(userEmb, itemIndices, app) {
        return tf.tidy(() => {
            const userTensor = tf.tensor2d([userEmb]);
            const scores = [];
            
            // Process items in smaller chunks to avoid memory issues
            for (let i = 0; i < itemIndices.length; i++) {
                const itemId = app.indexToItemId.get(itemIndices[i]);
                const itemFeatures = app.items.get(itemId).genres;
                const itemTensor = this.itemTower([itemFeatures]);
                
                const score = tf.matMul(userTensor, itemTensor, false, true).dataSync()[0];
                scores.push(score);
                
                itemTensor.dispose();
            }
            
            userTensor.dispose();
            return scores;
        });
    }

    async getItemEmbeddingsBatch(itemIndices, app) {
        return tf.tidy(() => {
            const itemFeaturesBatch = itemIndices.map(index => {
                const itemId = app.indexToItemId.get(index);
                return app.items.get(itemId).genres;
            });
            
            const embeddingsTensor = this.itemTower(itemFeaturesBatch);
            const embeddings = Array.from(embeddingsTensor.dataSync());
            
            // Convert flat array to 2D array
            const result = [];
            const embSize = this.embDim;
            for (let i = 0; i < embeddings.length; i += embSize) {
                result.push(embeddings.slice(i, i + embSize));
            }
            
            return result;
        });
    }
}
