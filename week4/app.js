class MovieLensApp {
    constructor() {
        this.interactions = [];
        this.items = new Map();
        this.users = new Map();
        this.userIdToIndex = new Map();
        this.itemIdToIndex = new Map();
        this.indexToUserId = new Map();
        this.indexToItemId = new Map();
        this.userRatedItems = new Map();
        this.model = null;
        this.isTraining = false;
        
        // Configuration
        this.config = {
            maxInteractions: 20000,
            epochs: 10,
            batchSize: 64,
            embeddingDim: 12,
            learningRate: 0.001,
            hiddenUnits: [32]
        };
        
        this.setupEventListeners();
    }

    setupEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.train());
        document.getElementById('test').addEventListener('click', () => this.test());
    }

    async loadData() {
        this.updateStatus('Loading data...');
        
        try {
            // Load and parse all three files in parallel
            const [interactionsData, itemsData, usersData] = await Promise.all([
                this.fetchFile('data/u.data'),
                this.fetchFile('data/u.item'),
                this.fetchFile('data/u.user')
            ]);

            this.parseItemsData(itemsData);
            this.parseUsersData(usersData);
            this.parseInteractionsData(interactionsData);
            
            this.updateStatus(`Data loaded: ${this.users.size} users, ${this.items.size} items, ${this.interactions.length} interactions`);
        } catch (error) {
            this.updateStatus(`Error loading data: ${error.message}`);
        }
    }

    async fetchFile(url) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to fetch ${url}`);
        return await response.text();
    }

    parseItemsData(data) {
        const lines = data.split('\n').filter(line => line.trim());
        
        for (const line of lines) {
            const parts = line.split('|');
            if (parts.length < 24) continue;
            
            const itemId = parseInt(parts[0]);
            const title = parts[1];
            const yearMatch = title.match(/\((\d{4})\)$/);
            const year = yearMatch ? parseInt(yearMatch[1]) : 0;
            
            // Parse genres (last 19 fields)
            const genres = parts.slice(5, 24).map(g => parseInt(g));
            
            this.items.set(itemId, {
                title: title.replace(/\(\d{4}\)$/, '').trim(),
                year,
                genres
            });
        }
    }

    parseUsersData(data) {
        const lines = data.split('\n').filter(line => line.trim());
        
        for (const line of lines) {
            const parts = line.split('|');
            if (parts.length < 5) continue;
            
            const userId = parseInt(parts[0]);
            const age = parseInt(parts[1]);
            const gender = parts[2];
            const occupation = parts[3];
            
            this.users.set(userId, { age, gender, occupation });
        }
    }

    parseInteractionsData(data) {
        const lines = data.split('\n').filter(line => line.trim());
        const allInteractions = [];
        
        for (const line of lines) {
            const parts = line.split('\t');
            if (parts.length < 4) continue;
            
            const userId = parseInt(parts[0]);
            const itemId = parseInt(parts[1]);
            const rating = parseFloat(parts[2]);
            const timestamp = parseInt(parts[3]);
            
            if (this.users.has(userId) && this.items.has(itemId)) {
                allInteractions.push({ userId, itemId, rating, timestamp });
            }
        }
        
        // Limit interactions for performance
        this.interactions = allInteractions
            .sort(() => Math.random() - 0.5)
            .slice(0, this.config.maxInteractions);
        
        this.buildIndexMappings();
        this.buildUserRatedItems();
    }

    buildIndexMappings() {
        // Build user index mappings
        const uniqueUserIds = [...new Set(this.interactions.map(i => i.userId))];
        uniqueUserIds.forEach((userId, index) => {
            this.userIdToIndex.set(userId, index);
            this.indexToUserId.set(index, userId);
        });

        // Build item index mappings
        const uniqueItemIds = [...new Set(this.interactions.map(i => i.itemId))];
        uniqueItemIds.forEach((itemId, index) => {
            this.itemIdToIndex.set(itemId, index);
            this.indexToItemId.set(index, itemId);
        });
    }

    buildUserRatedItems() {
        this.userRatedItems.clear();
        
        for (const interaction of this.interactions) {
            if (!this.userRatedItems.has(interaction.userId)) {
                this.userRatedItems.set(interaction.userId, []);
            }
            
            this.userRatedItems.get(interaction.userId).push({
                itemId: interaction.itemId,
                rating: interaction.rating,
                timestamp: interaction.timestamp
            });
        }
        
        // Sort each user's ratings by rating (desc) and timestamp (desc)
        for (const [userId, ratings] of this.userRatedItems) {
            ratings.sort((a, b) => {
                if (b.rating !== a.rating) return b.rating - a.rating;
                return b.timestamp - a.timestamp;
            });
        }
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
        console.log(message);
    }

    async train() {
        if (this.interactions.length === 0) {
            this.updateStatus('Please load data first');
            return;
        }

        this.isTraining = true;
        this.updateStatus('Initializing model...');
        
        // Initialize model
        const numUsers = this.userIdToIndex.size;
        const numItems = this.itemIdToIndex.size;
        const numGenres = 19; // Fixed for MovieLens
        const numOccupations = [...new Set([...this.users.values()].map(u => u.occupation))].length;
        const numGenders = 2; // M/F
        
        this.model = new TwoTowerModel(
            numUsers,
            numItems,
            this.config.embeddingDim,
            this.config.hiddenUnits,
            numGenres,
            numOccupations,
            numGenders
        );

        // Prepare training data
        const trainingPairs = this.interactions.map(interaction => ({
            userIndex: this.userIdToIndex.get(interaction.userId),
            itemIndex: this.itemIdToIndex.get(interaction.itemId)
        }));

        // Initialize loss chart
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        lossCtx.clearRect(0, 0, 600, 200);
        lossCtx.strokeStyle = 'blue';
        lossCtx.beginPath();
        lossCtx.moveTo(0, 200);

        // Training loop with yield to prevent blocking
        this.updateStatus('Starting training...');
        
        for (let epoch = 0; epoch < this.config.epochs; epoch++) {
            this.updateStatus(`Training epoch ${epoch + 1}/${this.config.epochs}`);
            
            // Shuffle training pairs
            const shuffled = [...trainingPairs].sort(() => Math.random() - 0.5);
            let epochLoss = 0;
            let batchCount = 0;

            for (let i = 0; i < shuffled.length; i += this.config.batchSize) {
                if (!this.isTraining) break;
                
                const batch = shuffled.slice(i, i + this.config.batchSize);
                const loss = await this.model.trainBatch(batch, this);
                
                epochLoss += loss;
                batchCount++;
                
                // Update loss chart
                const x = (i / shuffled.length + epoch) * (600 / this.config.epochs);
                const y = 200 - Math.min(loss * 10, 200);
                
                if (i === 0 && epoch === 0) {
                    lossCtx.moveTo(x, y);
                } else {
                    lossCtx.lineTo(x, y);
                }
                lossCtx.stroke();
                
                // Yield to prevent blocking
                if (i % (this.config.batchSize * 10) === 0) {
                    await new Promise(resolve => setTimeout(resolve, 0));
                }
            }
            
            const avgLoss = epochLoss / batchCount;
            this.updateStatus(`Epoch ${epoch + 1} completed. Average loss: ${avgLoss.toFixed(4)}`);
        }
        
        this.isTraining = false;
        this.updateStatus('Training completed!');
        this.visualizeEmbeddings();
    }

    async test() {
        if (!this.model) {
            this.updateStatus('Please train the model first');
            return;
        }

        // Find a user with at least 20 ratings
        const qualifiedUsers = [...this.userRatedItems.entries()]
            .filter(([_, ratings]) => ratings.length >= 20)
            .map(([userId]) => userId);
        
        if (qualifiedUsers.length === 0) {
            this.updateStatus('No users with sufficient ratings found');
            return;
        }
        
        const randomUserId = qualifiedUsers[Math.floor(Math.random() * qualifiedUsers.length)];
        const user = this.users.get(randomUserId);
        
        this.updateStatus(`Testing for user ${randomUserId} (${user.gender}, ${user.age}, ${user.occupation})`);
        
        // Get historical top 10
        const historicalTop10 = this.userRatedItems.get(randomUserId)
            .slice(0, 10)
            .map(rating => this.items.get(rating.itemId));
        
        // Get deep learning recommendations
        const dlTop10 = await this.getDLRecommendations(randomUserId);
        
        // Get content-based recommendations
        const cbTop10 = this.getContentBasedRecommendations(randomUserId);
        
        // Render comparison table
        this.renderComparisonTable(historicalTop10, dlTop10, cbTop10);
    }

    async getDLRecommendations(userId) {
        const userIndex = this.userIdToIndex.get(userId);
        const userData = this.users.get(userId);
        
        // Get user embedding
        const userEmb = await this.model.getUserEmbedding(userIndex, userData, this);
        
        // Score all items
        const allItemIndices = [...this.indexToItemId.keys()];
        const scores = await this.model.scoreUserItems(userEmb, allItemIndices, this);
        
        // Get top 10, excluding already rated items
        const ratedItemIds = new Set(this.userRatedItems.get(userId).map(r => r.itemId));
        
        const scoredItems = allItemIndices.map((itemIndex, i) => ({
            itemId: this.indexToItemId.get(itemIndex),
            score: scores[i]
        })).filter(item => !ratedItemIds.has(item.itemId))
          .sort((a, b) => b.score - a.score)
          .slice(0, 10)
          .map(item => this.items.get(item.itemId));
        
        return scoredItems;
    }

    getContentBasedRecommendations(userId) {
        // Get user's genre preferences from rated items
        const userRatings = this.userRatedItems.get(userId);
        const userGenreVector = new Array(19).fill(0);
        
        for (const rating of userRatings) {
            const item = this.items.get(rating.itemId);
            for (let i = 0; i < 19; i++) {
                userGenreVector[i] += item.genres[i] * rating.rating;
            }
        }
        
        // Normalize genre vector
        const magnitude = Math.sqrt(userGenreVector.reduce((sum, val) => sum + val * val, 0));
        const normalizedUserVector = userGenreVector.map(val => val / magnitude);
        
        // Score all items by cosine similarity, excluding rated items
        const ratedItemIds = new Set(userRatings.map(r => r.itemId));
        const scoredItems = [];
        
        for (const [itemId, item] of this.items) {
            if (ratedItemIds.has(itemId)) continue;
            
            const itemVector = item.genres;
            const itemMagnitude = Math.sqrt(itemVector.reduce((sum, val) => sum + val * val, 0));
            const normalizedItemVector = itemVector.map(val => val / itemMagnitude);
            
            // Cosine similarity
            const similarity = normalizedUserVector.reduce((sum, userVal, i) => 
                sum + userVal * normalizedItemVector[i], 0);
            
            scoredItems.push({ itemId, similarity, item });
        }
        
        return scoredItems
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, 10)
            .map(item => item.item);
    }

    renderComparisonTable(historical, dlRecs, cbRecs) {
        const maxLength = Math.max(historical.length, dlRecs.length, cbRecs.length);
        let html = `
            <div class="container">
                <div class="col">
                    <h3>Historical Top 10</h3>
                    <table>
                        <tr><th>Movie Title</th><th>Year</th></tr>
        `;
        
        // Historical column
        for (let i = 0; i < maxLength; i++) {
            html += '<tr>';
            if (i < historical.length) {
                html += `<td>${historical[i].title}</td><td>${historical[i].year}</td>`;
            } else {
                html += '<td></td><td></td>';
            }
            html += '</tr>';
        }
        
        html += `
                    </table>
                </div>
                <div class="col">
                    <h3>Deep Learning Recommendations</h3>
                    <table>
                        <tr><th>Movie Title</th><th>Year</th></tr>
        `;
        
        // DL recommendations column
        for (let i = 0; i < maxLength; i++) {
            html += '<tr>';
            if (i < dlRecs.length) {
                html += `<td>${dlRecs[i].title}</td><td>${dlRecs[i].year}</td>`;
            } else {
                html += '<td></td><td></td>';
            }
            html += '</tr>';
        }
        
        html += `
                    </table>
                </div>
                <div class="col">
                    <h3>Content-Based Recommendations</h3>
                    <table>
                        <tr><th>Movie Title</th><th>Year</th></tr>
        `;
        
        // Content-based recommendations column
        for (let i = 0; i < maxLength; i++) {
            html += '<tr>';
            if (i < cbRecs.length) {
                html += `<td>${cbRecs[i].title}</td><td>${cbRecs[i].year}</td>`;
            } else {
                html += '<td></td><td></td>';
            }
            html += '</tr>';
        }
        
        html += `
                    </table>
                </div>
            </div>
        `;
        
        document.getElementById('recommendations').innerHTML = html;
    }

    visualizeEmbeddings() {
        this.updateStatus('Computing PCA visualization...');
        
        // Sample items for visualization (for performance)
        const sampleItems = [...this.items.entries()]
            .sort(() => Math.random() - 0.5)
            .slice(0, 200);
        
        // Get item embeddings
        const itemIndices = sampleItems.map(([itemId]) => this.itemIdToIndex.get(itemId));
        const itemTensors = itemIndices.map(index => 
            tf.tensor1d(this.items.get(this.indexToItemId.get(index)).genres, 'float32')
        );
        
        // Get embeddings through item tower
        const embeddings = itemIndices.map((index, i) => {
            const emb = this.model.itemTower(itemTensors[i]).dataSync();
            itemTensors[i].dispose();
            return Array.from(emb);
        });
        
        // Simple PCA implementation
        const pcaResult = this.simplePCA(embeddings, 2);
        
        // Draw on canvas
        this.drawEmbeddings(pcaResult, sampleItems.map(([_, item]) => item));
        
        this.updateStatus('PCA visualization completed');
    }

    simplePCA(embeddings, targetDim) {
        // Center the data
        const mean = embeddings[0].map((_, i) => 
            embeddings.reduce((sum, emb) => sum + emb[i], 0) / embeddings.length
        );
        
        const centered = embeddings.map(emb => 
            emb.map((val, i) => val - mean[i])
        );
        
        // Compute covariance matrix (simplified)
        const cov = [];
        for (let i = 0; i < embeddings[0].length; i++) {
            cov[i] = [];
            for (let j = 0; j < embeddings[0].length; j++) {
                cov[i][j] = centered.reduce((sum, emb) => sum + emb[i] * emb[j], 0) / (embeddings.length - 1);
            }
        }
        
        // Simple power iteration for top 2 eigenvectors (approximation)
        let v1 = Array(embeddings[0].length).fill(1);
        let v2 = Array(embeddings[0].length).fill(1);
        
        // First component
        for (let iter = 0; iter < 10; iter++) {
            const newV1 = cov.map(row => 
                row.reduce((sum, val, j) => sum + val * v1[j], 0)
            );
            const norm = Math.sqrt(newV1.reduce((sum, val) => sum + val * val, 0));
            v1 = newV1.map(val => val / norm);
        }
        
        // Second component (orthogonal to first)
        for (let iter = 0; iter < 10; iter++) {
            const newV2 = cov.map(row => 
                row.reduce((sum, val, j) => sum + val * v2[j], 0)
            );
            // Make orthogonal to v1
            const dot = newV2.reduce((sum, val, i) => sum + val * v1[i], 0);
            const orthogonal = newV2.map((val, i) => val - dot * v1[i]);
            const norm = Math.sqrt(orthogonal.reduce((sum, val) => sum + val * val, 0));
            v2 = orthogonal.map(val => val / norm);
        }
        
        // Project data
        return embeddings.map(emb => ({
            x: emb.reduce((sum, val, i) => sum + val * v1[i], 0),
            y: emb.reduce((sum, val, i) => sum + val * v2[i], 0)
        }));
    }

    drawEmbeddings(points, items) {
        const canvas = document.getElementById('embeddingChart');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Find bounds
        const xs = points.map(p => p.x);
        const ys = points.map(p => p.y);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);
        
        // Scale points to canvas
        const scaleX = (canvas.width - 40) / (maxX - minX);
        const scaleY = (canvas.height - 40) / (maxY - minY);
        const scale = Math.min(scaleX, scaleY);
        
        const scaledPoints = points.map(p => ({
            x: 20 + (p.x - minX) * scale,
            y: canvas.height - 20 - (p.y - minY) * scale
        }));
        
        // Draw points
        ctx.fillStyle = 'rgba(0, 100, 255, 0.6)';
        for (let i = 0; i < scaledPoints.length; i++) {
            ctx.beginPath();
            ctx.arc(scaledPoints[i].x, scaledPoints[i].y, 3, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        // Add hover functionality
        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Find closest point
            let closestIndex = -1;
            let minDist = 10; // Only show tooltip if close enough
            
            for (let i = 0; i < scaledPoints.length; i++) {
                const dist = Math.sqrt(
                    Math.pow(x - scaledPoints[i].x, 2) + 
                    Math.pow(y - scaledPoints[i].y, 2)
                );
                if (dist < minDist) {
                    minDist = dist;
                    closestIndex = i;
                }
            }
            
            // Draw tooltip
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Redraw points
            for (let i = 0; i < scaledPoints.length; i++) {
                ctx.beginPath();
                ctx.arc(scaledPoints[i].x, scaledPoints[i].y, 3, 0, 2 * Math.PI);
                ctx.fill();
            }
            
            // Highlight closest point and show title
            if (closestIndex !== -1) {
                ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
                ctx.beginPath();
                ctx.arc(scaledPoints[closestIndex].x, scaledPoints[closestIndex].y, 5, 0, 2 * Math.PI);
                ctx.fill();
                
                ctx.fillStyle = 'black';
                ctx.font = '12px Arial';
                ctx.fillText(
                    items[closestIndex].title, 
                    scaledPoints[closestIndex].x + 8, 
                    scaledPoints[closestIndex].y - 8
                );
            }
        });
    }
}

// Initialize app when loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});
