let model;
let isTraining = false;

window.onload = async function() {
    try {
        // Load and parse data
        await loadData();
        
        // Populate dropdowns
        populateUserDropdown();
        populateMovieDropdown();
        
        // Start training
        await trainModel();
        
    } catch (error) {
        console.error('Initialization error:', error);
        document.getElementById('result').textContent = 
            'Error loading data or training model. Check console for details.';
        document.getElementById('result').className = 'result error';
    }
};

function populateUserDropdown() {
    const userSelect = document.getElementById('user-select');
    userSelect.innerHTML = '';
    
    for (let i = 1; i <= numUsers; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `User ${i}`;
        userSelect.appendChild(option);
    }
}

function populateMovieDropdown() {
    const movieSelect = document.getElementById('movie-select');
    movieSelect.innerHTML = '';
    
    movies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        movieSelect.appendChild(option);
    });
}

function createModel(numUsers, numMovies, latentDim = 10) {
    // User input and embedding
    const userInput = tf.input({ shape: [1], name: 'userInput' });
    const userEmbedding = tf.layers.embedding({
        inputDim: numUsers + 1, // +1 because user IDs start from 1
        outputDim: latentDim,
        name: 'userEmbedding'
    }).apply(userInput);
    const userVector = tf.layers.flatten().apply(userEmbedding);
    
    // Movie input and embedding
    const movieInput = tf.input({ shape: [1], name: 'movieInput' });
    const movieEmbedding = tf.layers.embedding({
        inputDim: numMovies + 1, // +1 because movie IDs start from 1
        outputDim: latentDim,
        name: 'movieEmbedding'
    }).apply(movieInput);
    const movieVector = tf.layers.flatten().apply(movieEmbedding);
    
    // Dot product for prediction
    const dotProduct = tf.layers.dot({ axes: -1 }).apply([userVector, movieVector]);
    
    // Create model
    const model = tf.model({
        inputs: [userInput, movieInput],
        outputs: dotProduct
    });
    
    return model;
}

async function trainModel() {
    try {
        isTraining = true;
        const resultDiv = document.getElementById('result');
        resultDiv.textContent = 'Training model... This may take a minute.';
        resultDiv.className = 'result loading';
        
        // Create model
        model = createModel(numUsers, numMovies, 10);
        
        // Compile model
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });
        
        // Prepare training data
        const userIds = ratings.map(r => r.userId);
        const movieIds = ratings.map(r => r.movieId);
        const ratingsValues = ratings.map(r => r.rating);
        
        const userTensor = tf.tensor2d(userIds, [userIds.length, 1]);
        const movieTensor = tf.tensor2d(movieIds, [movieIds.length, 1]);
        const ratingTensor = tf.tensor2d(ratingsValues, [ratingsValues.length, 1]);
        
        // Train model
        await model.fit([userTensor, movieTensor], ratingTensor, {
            epochs: 5,
            batchSize: 64,
            validationSplit: 0.1,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}`);
                    resultDiv.textContent = `Training... Epoch ${epoch + 1}/5, Loss: ${logs.loss.toFixed(4)}`;
                }
            }
        });
        
        // Clean up tensors
        tf.dispose([userTensor, movieTensor, ratingTensor]);
        
        // Update UI
        resultDiv.textContent = 'Model trained successfully! Ready for predictions.';
        resultDiv.className = 'result success';
        document.getElementById('predict-btn').disabled = false;
        isTraining = false;
        
    } catch (error) {
        console.error('Training error:', error);
        document.getElementById('result').textContent = 
            'Error training model. Check console for details.';
        document.getElementById('result').className = 'result error';
        isTraining = false;
    }
}

async function predictRating() {
    if (isTraining) {
        alert('Model is still training. Please wait...');
        return;
    }
    
    const userId = parseInt(document.getElementById('user-select').value);
    const movieId = parseInt(document.getElementById('movie-select').value);
    
    if (!userId || !movieId) {
        alert('Please select both a user and a movie.');
        return;
    }
    
    try {
        const resultDiv = document.getElementById('result');
        resultDiv.textContent = 'Making prediction...';
        resultDiv.className = 'result loading';
        
        // Create input tensors
        const userTensor = tf.tensor2d([[userId]]);
        const movieTensor = tf.tensor2d([[movieId]]);
        
        // Make prediction
        const prediction = model.predict([userTensor, movieTensor]);
        const rating = await prediction.data();
        const predictedRating = rating[0];
        
        // Clean up tensors
        tf.dispose([userTensor, movieTensor, prediction]);
        
        // Display result
        const selectedMovie = movies.find(m => m.id === movieId);
        resultDiv.innerHTML = `
            <strong>Predicted Rating for "${selectedMovie.title}"</strong><br>
            User ${userId} would rate this movie: <span style="font-size: 1.5em; color: #2c5530;">${predictedRating.toFixed(2)} / 5</span>
        `;
        resultDiv.className = 'result prediction';
        
    } catch (error) {
        console.error('Prediction error:', error);
        document.getElementById('result').textContent = 
            'Error making prediction. Check console for details.';
        document.getElementById('result').className = 'result error';
    }
}
