class SimpleMovieLensApp {
    constructor() {
        this.interactions = [];
        this.items = new Map();
        this.users = new Map();
        this.model = null;
        
        this.setupEventListeners();
        this.updateStatus("App loaded. Click Step 1.");
    }

    setupEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.trainModel());
        document.getElementById('test').addEventListener('click', () => this.testModel());
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
        console.log(message);
    }

    async loadData() {
        this.updateStatus("Loading data...");
        
        try {
            // Load MovieLens data files
            const [interactionsData, itemsData, usersData] = await Promise.all([
                this.fetchFile('data/u.data'),
                this.fetchFile('data/u.item'), 
                this.fetchFile('data/u.user')
            ]);

            this.parseItems(itemsData);
            this.parseUsers(usersData);
            this.parseInteractions(interactionsData);
            
            this.updateStatus(`✅ Data loaded: ${this.users.size} users, ${this.items.size} movies, ${this.interactions.length} ratings`);
            
        } catch (error) {
            this.updateStatus(`❌ Error: ${error.message}`);
        }
    }

    async fetchFile(url) {
        console.log(`Fetching ${url}...`);
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to load ${url}`);
        return await response.text();
    }

    parseItems(data) {
        const lines = data.split('\n');
        for (const line of lines) {
            const parts = line.split('|');
            if (parts.length < 5) continue;
            
            const itemId = parseInt(parts[0]);
            const title = parts[1];
            
            this.items.set(itemId, {
                title: title,
                genres: parts.slice(5, 24).map(g => parseInt(g))
            });
        }
        console.log(`Parsed ${this.items.size} items`);
    }

    parseUsers(data) {
        const lines = data.split('\n');
        for (const line of lines) {
            const parts = line.split('|');
            if (parts.length < 5) continue;
            
            const userId = parseInt(parts[0]);
            this.users.set(userId, {
                age: parseInt(parts[1]),
                gender: parts[2],
                occupation: parts[3]
            });
        }
        console.log(`Parsed ${this.users.size} users`);
    }

    parseInteractions(data) {
        const lines = data.split('\n');
        for (const line of lines) {
            const parts = line.split('\t');
            if (parts.length < 4) continue;
            
            const userId = parseInt(parts[0]);
            const itemId = parseInt(parts[1]);
            const rating = parseFloat(parts[2]);
            
            if (this.users.has(userId) && this.items.has(itemId)) {
                this.interactions.push({ userId, itemId, rating });
            }
        }
        
        // Use only first 1000 interactions for speed
        this.interactions = this.interactions.slice(0, 1000);
        console.log(`Parsed ${this.interactions.length} interactions`);
    }

    async trainModel() {
        if (this.interactions.length === 0) {
            this.updateStatus("❌ Please load data first");
            return;
        }

        this.updateStatus("Training simple model...");
        
        try {
            // Create user and item indices
            const userIndices = [...new Set(this.interactions.map(i => i.userId))];
            const itemIndices = [...new Set(this.interactions.map(i => i.itemId))];
            
            const userToIndex = new Map(userIndices.map((id, idx) => [id, idx]));
            const itemToIndex = new Map(itemIndices.map((id, idx) => [id, idx]));
            
            // Simple embedding model
            const numUsers = userIndices.length;
            const numItems = itemIndices.length;
            const embeddingDim = 8;
            
            // User embeddings
            const userEmbeddings = tf.variable(
                tf.randomNormal([numUsers, embeddingDim]), 
                true, 
                'user_embeddings'
            );
            
            // Item embeddings  
            const itemEmbeddings = tf.variable(
                tf.randomNormal([numItems, embeddingDim]),
                true,
                'item_embeddings'
            );
            
            const optimizer = tf.train.adam(0.01);
            
            // Train for a few epochs
            for (let epoch = 0; epoch < 3; epoch++) {
                let totalLoss = 0;
                let batchCount = 0;
                
                // Use small batches
                for (let i = 0; i < this.interactions.length; i += 32) {
                    const batch = this.interactions.slice(i, i + 32);
                    
                    const loss = optimizer.minimize(() => {
                        let batchLoss = tf.scalar(0);
                        
                        for (const interaction of batch) {
                            const userIdx = userToIndex.get(interaction.userId);
                            const itemIdx = itemToIndex.get(interaction.itemId);
                            
                            if (userIdx === undefined || itemIdx === undefined) continue;
                            
                            const userEmb = userEmbeddings.slice([userIdx, 0], [1, embeddingDim]);
                            const itemEmb = itemEmbeddings.slice([itemIdx, 0], [1, embeddingDim]);
                            
                            // Dot product prediction
                            const prediction = tf.sum(tf.mul(userEmb, itemEmb));
                            const target = tf.scalar(interaction.rating / 5.0); // Normalize rating
                            
                            // MSE loss
                            const diff = tf.sub(prediction, target);
                            batchLoss = tf.add(batchLoss, tf.mul(diff, diff));
                        }
                        
                        return tf.div(batchLoss, tf.scalar(batch.length));
                    }, true);
                    
                    if (loss) {
                        totalLoss += loss.dataSync()[0];
                        loss.dispose();
                    }
                    
                    batchCount++;
                    
                    // Prevent blocking
                    if (batchCount % 5 === 0) {
                        await new Promise(resolve => setTimeout(resolve, 10));
                    }
                }
                
                const avgLoss = totalLoss / batchCount;
                this.updateStatus(`Epoch ${epoch + 1}/3 completed. Loss: ${avgLoss.toFixed(4)}`);
            }
            
            // Store the simple model
            this.model = { userEmbeddings, itemEmbeddings, userToIndex, itemToIndex };
            this.updateStatus("✅ Model training completed!");
            
        } catch (error) {
            this.updateStatus(`❌ Training error: ${error.message}`);
            console.error(error);
        }
    }

    async testModel() {
        if (!this.model) {
            this.updateStatus("❌ Please train model first");
            return;
        }

        this.updateStatus("Generating recommendations...");
        
        try {
            // Find a random user with some ratings
            const userRatings = new Map();
            this.interactions.forEach(interaction => {
                if (!userRatings.has(interaction.userId)) {
                    userRatings.set(interaction.userId, []);
                }
                userRatings.get(interaction.userId).push(interaction);
            });
            
            const usersWithRatings = [...userRatings.entries()].filter(([_, ratings]) => ratings.length >= 5);
            if (usersWithRatings.length === 0) {
                this.updateStatus("❌ No users with enough ratings found");
                return;
            }
            
            const [userId, ratings] = usersWithRatings[Math.floor(Math.random() * usersWithRatings.length)];
            const user = this.users.get(userId);
            
            this.updateStatus(`Testing for user ${userId} (${user.gender}, ${user.age}, ${user.occupation})`);
            
            // Get user embedding
            const userIdx = this.model.userToIndex.get(userId);
            if (userIdx === undefined) {
                this.updateStatus("❌ User not found in model");
                return;
            }
            
            const userEmb = this.model.userEmbeddings.slice([userIdx, 0], [1, 8]);
            
            // Score all items
            const ratedItemIds = new Set(ratings.map(r => r.itemId));
            const scores = [];
            
            for (const [itemId, item] of this.items) {
                if (ratedItemIds.has(itemId)) continue;
                
                const itemIdx = this.model.itemToIndex.get(itemId);
                if (itemIdx === undefined) continue;
                
                const itemEmb = this.model.itemEmbeddings.slice([itemIdx, 0], [1, 8]);
                const score = tf.sum(tf.mul(userEmb, itemEmb)).dataSync()[0];
                
                scores.push({ itemId, score, item });
                
                itemEmb.dispose();
            }
            
            userEmb.dispose();
            
            // Get top 10 recommendations
            const recommendations = scores
                .sort((a, b) => b.score - a.score)
                .slice(0, 10)
                .map(item => item.item);
            
            // Display results
            this.displayResults(ratings.slice(0, 10), recommendations, user);
            
        } catch (error) {
            this.updateStatus(`❌ Test error: ${error.message}`);
            console.error(error);
        }
    }

    displayResults(ratings, recommendations, user) {
        let html = `<h2>Recommendations for User (${user.gender}, ${user.age}, ${user.occupation})</h2>`;
        
        html += `<div style="display: flex; gap: 20px;">`;
        
        // Top rated movies
        html += `<div style="flex: 1;">`;
        html += `<h3>User's Top Rated Movies</h3>`;
        html += `<table><tr><th>Movie</th><th>Rating</th></tr>`;
        
        ratings.sort((a, b) => b.rating - a.rating);
        ratings.forEach(rating => {
            const movie = this.items.get(rating.itemId);
            html += `<tr><td>${movie.title}</td><td>${rating.rating}/5</td></tr>`;
        });
        html += `</table></div>`;
        
        // Recommendations
        html += `<div style="flex: 1;">`;
        html += `<h3>Recommended Movies</h3>`;
        html += `<table><tr><th>Movie</th><th>Score</th></tr>`;
        
        recommendations.forEach(rec => {
            html += `<tr><td>${rec.title}</td><td>${rec.score.toFixed(3)}</td></tr>`;
        });
        html += `</table></div>`;
        
        html += `</div>`;
        
        document.getElementById('output').innerHTML = html;
        this.updateStatus("✅ Recommendations generated!");
    }
}

// Start app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new SimpleMovieLensApp();
});
