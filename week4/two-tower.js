class TwoTowerModel {
    constructor(numUsers, numItems, numGenres, numOccupations, numGenders, config) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.numGenres = numGenres;
        this.numOccupations = numOccupations;
        this.numGenders = numGenders;
        this.config = config;
        
        this.initializeModel();
        this.initializeOptimizer();
    }

    initializeModel() {
        const { embeddingDim, hiddenUnits } = this.config;
        
        // User tower components
        this.userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: embeddingDim,
            inputLength: 1,
            name: 'user_embedding'
        });
        
        this.ageNormalization = tf.layers.dense({
            units: 1,
            activation: 'linear',
            useBias: true,
            name: 'age_normalization'
        });
        
        this.genderEmbedding = tf.layers.embedding({
            inputDim: this.numGenders,
            outputDim: Math.floor(embeddingDim / 4),
            inputLength: 1,
            name: 'gender_embedding'
        });
        
        this.occupationEmbedding = tf.layers.embedding({
            inputDim: this.numOccupations,
            outputDim: Math.floor(embeddingDim / 2),
            inputLength: 1,
            name: 'occupation_embedding'
        });
        
        // User MLP
        this.userMLP = [];
        hiddenUnits.forEach((units, i) => {
            this.userMLP.push(tf.layers.dense({
                units: units,
                activation: 'relu',
                name: `user_mlp_${i}`
            }));
        });
        this.userMLP.push(tf.layers.dense({
            units: embeddingDim,
            activation: 'linear',
            name: 'user_output'
        }));
        
        // Item tower components
        this.itemEmbedding = tf.layers.embedding({
            inputDim: this.numItems,
            outputDim: embeddingDim,
            inputLength: 1,
            name: 'item_embedding'
        });
        
        this.genreProjection = tf.layers.dense({
            units: embeddingDim,
            activation: 'linear',
            useBias: true,
            name: 'genre_projection'
        });
        
        // Item MLP
        this.itemMLP = [];
        hiddenUnits.forEach((units, i) => {
            this.itemMLP.push(tf.layers.dense({
                units: units,
                activation: 'relu',
                name: `item_mlp_${i}`
            }));
        });
        this.itemMLP.push(tf.layers.dense({
            units: embeddingDim,
            activation: 'linear',
            name: 'item_output'
        }));
        
        // Create occupation and gender mappings
        this.occupationMap = this.createOccupationMap();
        this.genderMap = { 'M': 0, 'F': 1 };
    }

    createOccupationMap() {
        // This would normally come from the data, using placeholder
        const occupations = [
            'educator', 'engineer', 'healthcare', 'artist', 'executive', 
            'lawyer', 'librarian', 'marketing', 'none', 'other', 
            'programmer', 'retired', 'salesman', 'scientist', 'student', 
            'technician', 'writer', 'homemaker', 'doctor', 'entertainment'
        ];
        const map = {};
        occupations.forEach((occ, idx) => map[occ] = idx);
        return map;
    }

    initializeOptimizer() {
        this.optimizer = tf.train.adam(this.config.learningRate);
    }

    userForward(userIndices, userFeatures) {
        // userIndices: [batchSize, 1]
        // userFeatures: {ages: [batchSize, 1], genders: [batchSize, 1], occupations: [batchSize, 1]}
        
        let embedding = this.userEmbedding.apply(userIndices);
        embedding = tf.squeeze(embedding, [1]);
        
        // Process age
        const ageNormalized = this.ageNormalization.apply(userFeatures.ages);
        
        // Process gender
        const genderEmbedding = this.genderEmbedding.apply(userFeatures.genders);
        const genderSqueezed = tf.squeeze(genderEmbedding, [1]);
        
        // Process occupation
        const occupationEmbedding = this.occupationEmbedding.apply(userFeatures.occupations);
        const occupationSqueezed = tf.squeeze(occupationEmbedding, [1]);
        
        // Concatenate all user features
        let userRep = tf.concat([embedding, ageNormalized, genderSqueezed, occupationSqueezed], -1);
        
        // Apply MLP
        for (const layer of this.userMLP) {
            userRep = layer.apply(userRep);
        }
        
        // L2 normalize
        return tf.l2Normalize(userRep, -1);
    }

    itemForward(itemIndices, genreFeatures) {
        // itemIndices: [batchSize, 1]
        // genreFeatures: [batchSize, numGenres]
        
        let embedding = this.itemEmbedding.apply(itemIndices);
        embedding = tf.squeeze(embedding, [1]);
        
        // Process genres
        const genreProjected = this.genreProjection.apply(genreFeatures);
        
        // Combine item embedding and genre features
        let itemRep = tf.add(embedding, genreProjected);
        
        // Apply MLP
        for (const layer of this.itemMLP) {
            itemRep = layer.apply(itemRep);
        }
        
        // L2 normalize
        return tf.l2Normalize(itemRep, -1);
    }

    score(userEmbeddings, itemEmbeddings) {
        // Dot product similarity
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1, true);
    }

    computeLoss(userEmbeddings, itemEmbeddings) {
        // In-batch sampled softmax loss
        const batchSize = userEmbeddings.shape[0];
        const embeddingDim = userEmbeddings.shape[1];
        
        // Compute scores: [batchSize, batchSize]
        const scores = tf.matMul(userEmbeddings, itemEmbeddings, false, true);
        
        // Labels are diagonal (each user matches with corresponding item)
        const labels = tf.oneHot(tf.range(0, batchSize), batchSize);
        
        // Softmax cross entropy loss
        const loss = tf.losses.softmaxCrossEntropy(labels, scores);
        
        return loss;
    }

    async trainEpoch(interactions, userIdMap, itemIdMap, users, items) {
        const { batchSize, maxInteractions } = this.config;
        const numBatches = Math.ceil(Math.min(interactions.length, maxInteractions) / batchSize);
        let totalLoss = 0;
        
        // Shuffle interactions
        const shuffled = [...interactions].sort(() => Math.random() - 0.5);
        const trainingInteractions = shuffled.slice(0, maxInteractions);
        
        for (let batch = 0; batch < numBatches; batch++) {
            const start = batch * batchSize;
            const end = Math.min(start + batchSize, trainingInteractions.length);
            const batchInteractions = trainingInteractions.slice(start, end);
            
            const loss = await this.trainBatch(batchInteractions, userIdMap, itemIdMap, users, items);
            totalLoss += loss;
            
            // Clean up memory
            tf.disposeVariables();
        }
        
        return totalLoss / numBatches;
    }

    async trainBatch(interactions, userIdMap, itemIdMap, users, items) {
        return tf.tidy(() => {
            const batchSize = interactions.length;
            
            // Prepare batch data
            const userIndices = [];
            const itemIndices = [];
            const userAges = [];
            const userGenders = [];
            const userOccupations = [];
            const itemGenres = [];
            
            interactions.forEach(interaction => {
                const userId = interaction.userId;
                const itemId = interaction.itemId;
                const userData = users.get(userId);
                
                userIndices.push(userIdMap.get(userId));
                itemIndices.push(itemIdMap.get(itemId));
                userAges.push(userData.age);
                userGenders.push(this.genderMap[userData.gender] || 0);
                userOccupations.push(this.occupationMap[userData.occupation] || 0);
                itemGenres.push(items.get(itemId).genres);
            });
            
            // Convert to tensors
            const userIndicesTensor = tf.tensor2d(userIndices, [batchSize, 1], 'int32');
            const itemIndicesTensor = tf.tensor2d(itemIndices, [batchSize, 1], 'int32');
            const userAgesTensor = tf.tensor2d(userAges, [batchSize, 1], 'float32');
            const userGendersTensor = tf.tensor2d(userGenders, [batchSize, 1], 'int32');
            const userOccupationsTensor = tf.tensor2d(userOccupations, [batchSize, 1], 'int32');
            const itemGenresTensor = tf.tensor2d(itemGenres, [batchSize, this.numGenres], 'float32');
            
            const userFeatures = {
                ages: userAgesTensor,
                genders: userGendersTensor,
                occupations: userOccupationsTensor
            };
            
            const loss = this.optimizer.minimize(() => {
                const userEmbeddings = this.userForward(userIndicesTensor, userFeatures);
                const itemEmbeddings = this.itemForward(itemIndicesTensor, itemGenresTensor);
                
                return this.computeLoss(userEmbeddings, itemEmbeddings);
            }, true);
            
            return loss ? loss.dataSync()[0] : 0;
        });
    }

    async getUserEmbedding(userIdx, userData) {
        return tf.tidy(() => {
            const userIndices = tf.tensor2d([[userIdx]], [1, 1], 'int32');
            const userAges = tf.tensor2d([[userData.age]], [1, 1], 'float32');
            const userGenders = tf.tensor2d([[this.genderMap[userData.gender] || 0]], [1, 1], 'int32');
            const userOccupations = tf.tensor2d([[this.occupationMap[userData.occupation] || 0]], [1, 1], 'int32');
            
            const userFeatures = {
                ages: userAges,
                genders: userGenders,
                occupations: userOccupations
            };
            
            return this.userForward(userIndices, userFeatures);
        });
    }

    async getItemEmbedding(itemIdx, genres) {
        return tf.tidy(() => {
            const itemIndices = tf.tensor2d([[itemIdx]], [1, 1], 'int32');
            const genreFeatures = tf.tensor2d([genres], [1, this.numGenres], 'float32');
            
            return this.itemForward(itemIndices, genreFeatures);
        });
    }

    async scoreUserItem(userEmbedding, itemIdx, genres) {
        return tf.tidy(() => {
            const itemEmbedding = this.getItemEmbedding(itemIdx, genres);
            const score = this.score(userEmbedding, itemEmbedding);
            return score.dataSync()[0];
        });
    }
}
