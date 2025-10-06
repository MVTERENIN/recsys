class TwoTowerModel {
    constructor(numUsers, numItems, embDim, hiddenUnits, numGenres, numOccupations, numGenders) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embDim = embDim;
        this.hiddenUnits = hiddenUnits;
        
        // Feature dimensions
        this.numGenres = numGenres;
        this.numOccupations = numOccupations;
        this.numGenders = numGenders;
        
        // Create embedding layers and MLPs
        this.createUserTower();
        this.createItemTower();
        
        // Optimizer
        this.optimizer = tf.train.adam(0.001);
        
        // Lookup tables for categorical features
        this.genderToIndex = { 'M': 0, 'F': 1 };
        this.occupationToIndex = {
            'administrator': 0, 'artist': 1, 'doctor': 2, 'educator': 3, 'engineer': 4,
            'entertainment': 5, 'executive': 6, 'healthcare': 7, 'homemaker': 8, 'lawyer': 9,
            'librarian': 10, 'marketing': 11, 'none': 12, 'other': 13, 'programmer': 14,
            'retired': 15, 'salesman': 16, 'scientist': 17, 'student': 18, 'technician': 19, 'writer': 20
        };
    }

    createUserTower() {
        // User feature embeddings
        this.userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: this.embDim,
            inputLength: 1,
            name: 'user_embedding'
        });
        
        // Age normalization (assuming max age ~100)
        this.ageNormalization = tf.layers.dense({
            units: 1,
            activation: 'linear',
            useBias: false,
            name: 'age_norm'
        });
        
        // Gender embedding
        this.genderEmbedding = tf.layers.embedding({
            inputDim: this.numGenders,
            outputDim: Math.floor(this.embDim / 4),
            inputLength: 1,
            name: 'gender_embedding'
        });
        
        // Occupation embedding
        this.occupationEmbedding = tf.layers.embedding({
            inputDim: this.numOccupations,
            outputDim: Math.floor(this.embDim / 2),
            inputLength: 1,
            name: 'occupation_embedding'
        });
        
        // User MLP
        this.userMLP = tf.sequential({
            layers: [
                tf.layers.dense({
                    units: this.hiddenUnits,
                    activation: 'relu',
                    name: 'user_mlp_hidden'
                }),
                tf.layers.dense({
                    units: this.embDim,
                    activation: 'linear',
                    name: 'user_mlp_output'
                })
            ]
        });
    }

    createItemTower() {
        // Item embedding (for genres)
        this.itemEmbedding = tf.layers.embedding({
            inputDim: this.numItems,
            outputDim: this.embDim,
            inputLength: 1,
            name: 'item_embedding'
        });
        
        // Genre MLP (processes genre vectors)
        this.genreMLP = tf.sequential({
            layers: [
                tf.layers.dense({
                    units: this.hiddenUnits,
                    activation: 'relu',
                    name: 'genre_mlp_hidden'
                }),
                tf.layers.dense({
                    units: this.embDim,
                    activation: 'linear',
                    name: 'genre_mlp_output'
                })
            ]
        });
        
        // Alternative: Direct genre processing (since genres are already vectors)
        this.genreProjection = tf.layers.dense({
            units: this.embDim,
            activation: 'relu',
            name: 'genre_projection'
        });
    }

    userForward(userIndices, userFeatures) {
        // Process user through user tower
        const userEmb = this.userEmbedding.apply(tf.tensor1d(userIndices, 'int32'));
        
        // Process additional features
        const ageTensor = tf.tensor2d(userFeatures.map(f => [f.age / 100.0])); // Normalize age
        const normalizedAge = this.ageNormalization.apply(ageTensor);
        
        const genderTensor = tf.tensor1d(userFeatures.map(f => this.genderToIndex[f.gender] || 0), 'int32');
        const genderEmb = this.genderEmbedding.apply(genderTensor);
        
        const occupationTensor = tf.tensor1d(userFeatures.map(f => this.occupationToIndex[f.occupation] || 0), 'int32');
        const occupationEmb = this.occupationEmbedding.apply(occupationTensor);
        
        // Concatenate all user features
        const userFeaturesConcat = tf.concat([
            userEmb,
            normalizedAge,
            genderEmb,
            occupationEmb
        ], 1);
        
        // Apply MLP
        return this.userMLP.apply(userFeaturesConcat);
    }

    itemForward(itemIndices, itemFeatures) {
        // Two approaches: use item embedding or genre-based features
        // Using genre-based features for better content understanding
        const genreTensor = tf.tensor2d(itemFeatures.map(f => f.genres));
        return this.genreProjection.apply(genreTensor);
    }

    score(userEmbeddings, itemEmbeddings) {
        // Dot product between user and item embeddings
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), 1);
    }

    async trainStep(userIndices, itemIndices, userFeatures, itemFeatures) {
        return tf.tidy(() => {
            const userEmbs = this.userForward(userIndices, userFeatures);
            const itemEmbs = this.itemForward(itemIndices, itemFeatures);
            
            // In-batch sampled softmax loss
            const logits = tf.matMul(userEmbs, itemEmbs, false, true); // U @ I^T
            const labels = tf.oneHot(tf.range(0, userIndices.length), userIndices.length);
            
            const loss = tf.losses.softmaxCrossEntropy(labels, logits);
            
            // Compute gradients and update weights
            const variables = this.getTrainableVariables();
            const gradients = tf.grad(loss => loss)(variables);
            
            this.optimizer.applyGradients(gradients.map((grad, i) => ({
                gradient: grad,
                variable: variables[i]
            })));
            
            return loss.dataSync()[0];
        });
    }

    getTrainableVariables() {
        const variables = [];
        
        // User tower variables
        variables.push(...this.userEmbedding.trainableWeights);
        variables.push(...this.ageNormalization.trainableWeights);
        variables.push(...this.genderEmbedding.trainableWeights);
        variables.push(...this.occupationEmbedding.trainableWeights);
        variables.push(...this.userMLP.trainableWeights);
        
        // Item tower variables
        variables.push(...this.itemEmbedding.trainableWeights);
        variables.push(...this.genreProjection.trainableWeights);
        
        return variables;
    }

    async getUserEmbedding(userFeatures) {
        return tf.tidy(() => {
            // Create dummy indices for user embedding lookup
            const dummyIndices = new Array(userFeatures.length).fill(0);
            const embeddings = this.userForward(dummyIndices, userFeatures);
            return embeddings.arraySync();
        });
    }

    async getItemEmbeddings(itemFeatures) {
        return tf.tidy(() => {
            const dummyIndices = new Array(itemFeatures.length).fill(0);
            const embeddings = this.itemForward(dummyIndices, itemFeatures);
            return embeddings.arraySync();
        });
    }

    async getScoresForAllItems(userEmbedding, itemFeatures) {
        return tf.tidy(() => {
            const userEmbTensor = tf.tensor2d([userEmbedding]);
            const itemEmbs = this.itemForward(new Array(itemFeatures.length).fill(0), itemFeatures);
            
            const scores = tf.matMul(userEmbTensor, itemEmbs, false, true);
            return scores.dataSync();
        });
    }
}
