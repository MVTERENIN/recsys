class TwoTowerModel {
    constructor(numUsers, numItems, embDim, hiddenUnits, numGenres, numOccupations, numGenders) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embDim = embDim;
        this.hiddenUnits = hiddenUnits;
        this.numGenres = numGenres;
        this.numOccupations = numOccupations;
        this.numGenders = numGenders;
        
        this.optimizer = tf.train.adam(0.001);
        
        this.buildModel();
    }

    buildModel() {
        // User tower components
        this.userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: this.embDim,
            inputLength: 1,
            name: 'user_embedding'
        });

        this.ageNormalization = tf.layers.dense({
            units: 1,
            activation: 'linear',
            useBias: true,
            name: 'age_norm'
        });

        this.genderEmbedding = tf.layers.embedding({
            inputDim: this.numGenders,
            outputDim: Math.floor(this.embDim / 4),
            inputLength: 1,
            name: 'gender_embedding'
        });

        this.occupationEmbedding = tf.layers.embedding({
            inputDim: this.numOccupations,
            outputDim: Math.floor(this.embDim / 2),
            inputLength: 1,
            name: 'occupation_embedding'
        });

        // Item tower component
        this.itemEmbedding = tf.layers.embedding({
            inputDim: this.numItems,
            outputDim: this.embDim,
            inputLength: 1,
            name: 'item_embedding'
        });

        this.genreProjection = tf.layers.dense({
            units: this.embDim,
            activation: 'linear',
            useBias: true,
            name: 'genre_projection'
        });

        // MLP layers for both towers
        this.userMLP = [];
        this.itemMLP = [];
        
        for (const units of this.hiddenUnits) {
            this.userMLP.push(
                tf.layers.dense({
                    units: units,
                    activation: 'relu',
                    name: 'user_mlp_' + units
                })
            );
            
            this.itemMLP.push(
                tf.layers.dense({
                    units: units,
                    activation: 'relu',
                    name: 'item_mlp_' + units
                })
            );
        }
        
        // Final embedding layers
        this.userFinal = tf.layers.dense({
            units: this.embDim,
            activation: 'linear',
            name: 'user_final'
        });

        this.itemFinal = tf.layers.dense({
            units: this.embDim,
            activation: 'linear',
            name: 'item_final'
        });
    }

    userTower(userIdTensor, userFeatures, app) {
        // User ID embedding
        const userEmb = this.userEmbedding.apply(userIdTensor);
        
        // Process demographic features
        const ageTensor = tf.tensor1d([userFeatures.age / 100.0], 'float32'); // Normalize age
        const ageNorm = this.ageNormalization.apply(ageTensor);
        
        const genderIndex = userFeatures.gender === 'M' ? 0 : 1;
        const genderTensor = tf.tensor1d([genderIndex], 'int32');
        const genderEmb = this.genderEmbedding.apply(genderTensor);
        
        // Occupation mapping
        const occupations = [...new Set([...app.users.values()].map(u => u.occupation))];
        const occupationIndex = occupations.indexOf(userFeatures.occupation);
        const occupationTensor = tf.tensor1d([occupationIndex], 'int32');
        const occupationEmb = this.occupationEmbedding.apply(occupationTensor);
        
        // Concatenate all user features
        let userFeaturesCombined = tf.concat([
            userEmb.flatten(),
            ageNorm.flatten(),
            genderEmb.flatten(),
            occupationEmb.flatten()
        ], -1);
        
        userFeaturesCombined = userFeaturesCombined.expandDims(0);
        
        // Apply MLP
        for (const layer of this.userMLP) {
            userFeaturesCombined = layer.apply(userFeaturesCombined);
        }
        
        // Final projection
        const userOutput = this.userFinal.apply(userFeaturesCombined);
        return userOutput;
    }

    itemTower(itemFeatures) {
        // Project genre features
        let itemFeaturesTensor = itemFeatures.expandDims(0);
        itemFeaturesTensor = this.genreProjection.apply(itemFeaturesTensor);
        
        // Apply MLP
        for (const layer of this.itemMLP) {
            itemFeaturesTensor = layer.apply(itemFeaturesTensor);
        }
        
        // Final projection
        const itemOutput = this.itemFinal.apply(itemFeaturesTensor);
        return itemOutput;
    }

    async trainBatch(batchPairs, app) {
        return tf.tidy(() => {
            const userIds = batchPairs.map(pair => pair.userIndex);
            const itemIds = batchPairs.map(pair => pair.itemIndex);
            
            const userTensors = userIds.map(userIndex => {
                const userId = app.indexToUserId.get(userIndex);
                const userFeatures = app.users.get(userId);
                return this.userTower(
                    tf.tensor1d([userIndex], 'int32'), 
                    userFeatures, 
                    app
                );
            });
            
            const itemTensors = itemIds.map(itemIndex => {
                const itemId = app.indexToItemId.get(itemIndex);
                const itemFeatures = app.items.get(itemId);
                return this.itemTower(
                    tf.tensor1d(itemFeatures.genres, 'float32')
                );
            });
            
            // Stack all tensors
            const userEmbeddings = tf.stack(userTensors.map(t => t.squeeze()));
            const itemEmbeddings = tf.stack(itemTensors.map(t => t.squeeze()));
            
            // Clean up intermediate tensors
            userTensors.forEach(t => t.dispose());
            itemTensors.forEach(t => t.dispose());
            
            // Compute scores using dot product
            const scores = tf.matMul(userEmbeddings, itemEmbeddings, false, true);
            
            // In-batch sampled softmax loss
            const batchSize = batchPairs.length;
            const labels = tf.oneHot(tf.range(0, batchSize), batchSize);
            
            const loss = tf.losses.softmaxCrossEntropy(labels, scores).mean();
            
            // Optimization
            const variables = this.getAllVariables();
            const grads = tf.grad(loss => loss).call(this, loss, variables);
            this.optimizer.applyGradients(grads.map((grad, i) => ({
                gradient: grad,
                variable: variables[i]
            })));
            
            // Clean up
            variables.forEach(v => v.dispose());
            grads.forEach(g => g.dispose());
            userEmbeddings.dispose();
            itemEmbeddings.dispose();
            scores.dispose();
            labels.dispose();
            
            return loss.dataSync()[0];
        });
    }

    getAllVariables() {
        const variables = [];
        
        // Collect all layer variables
        const layers = [
            this.userEmbedding, this.ageNormalization, this.genderEmbedding, 
            this.occupationEmbedding, this.itemEmbedding, this.genreProjection,
            ...this.userMLP, ...this.itemMLP, this.userFinal, this.itemFinal
        ];
        
        for (const layer of layers) {
            if (layer.trainableWeights) {
                variables.push(...layer.trainableWeights);
            }
        }
        
        return variables;
    }

    async getUserEmbedding(userIndex, userFeatures, app) {
        return tf.tidy(() => {
            const userTensor = this.userTower(
                tf.tensor1d([userIndex], 'int32'),
                userFeatures,
                app
            );
            const embedding = userTensor.dataSync();
            userTensor.dispose();
            return Array.from(embedding);
        });
    }

    async scoreUserItems(userEmb, itemIndices, app) {
        return tf.tidy(() => {
            const userTensor = tf.tensor2d([userEmb]);
            
            // Batch process items to avoid memory issues
            const batchSize = 100;
            const allScores = [];
            
            for (let i = 0; i < itemIndices.length; i += batchSize) {
                const batchIndices = itemIndices.slice(i, i + batchSize);
                
                const itemTensors = batchIndices.map(itemIndex => {
                    const itemId = app.indexToItemId.get(itemIndex);
                    const itemFeatures = app.items.get(itemId);
                    return this.itemTower(
                        tf.tensor1d(itemFeatures.genres, 'float32')
                    );
                });
                
                const itemEmbeddings = tf.stack(itemTensors.map(t => t.squeeze()));
                const batchScores = tf.matMul(userTensor, itemEmbeddings, false, true);
                
                allScores.push(...batchScores.dataSync());
                
                // Clean up
                itemTensors.forEach(t => t.dispose());
                itemEmbeddings.dispose();
                batchScores.dispose();
            }
            
            userTensor.dispose();
            return allScores;
        });
    }
}
