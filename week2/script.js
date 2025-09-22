// Initialize the application when the window loads
window.onload = async function() {
    try {
        // Display loading message
        const resultElement = document.getElementById('result');
        resultElement.textContent = "Loading movie data...";
        resultElement.className = 'loading';
        
        // Load data
        await loadData();
        
        // Populate dropdown and update status
        populateMoviesDropdown();
        resultElement.textContent = "Data loaded. Please select a movie.";
        resultElement.className = 'success';
    } catch (error) {
        console.error('Initialization error:', error);
        // Error message already set in data.js
    }
};

// Populate the movies dropdown with sorted movie titles
function populateMoviesDropdown() {
    const selectElement = document.getElementById('movie-select');
    
    // Clear existing options except the first placeholder
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }
    
    // Sort movies alphabetically by title
    const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));
    
    // Add movies to dropdown
    sortedMovies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        selectElement.appendChild(option);
    });
}

// Main recommendation function

function getRecommendations() {
    const resultElement = document.getElementById('result');
    
    try {
        // Step 1: Get user input
        const selectElement = document.getElementById('movie-select');
        const selectedMovieId = parseInt(selectElement.value);
        
        if (isNaN(selectedMovieId)) {
            resultElement.textContent = "Please select a movie first.";
            resultElement.className = 'error';
            return;
        }
        
        // Step 2: Find the liked movie
        const likedMovie = movies.find(movie => movie.id === selectedMovieId);
        if (!likedMovie) {
            resultElement.textContent = "Error: Selected movie not found in database.";
            resultElement.className = 'error';
            return;
        }
        
        // Show loading message while processing
        resultElement.textContent = "Calculating recommendations...";
        resultElement.className = 'loading';
        
        // Use setTimeout to allow the UI to update before heavy computation
        setTimeout(() => {
            try {
                // Step 3: Get all unique genres across all movies
                const allGenres = new Set();
                movies.forEach(movie => {
                    movie.genres.forEach(genre => allGenres.add(genre));
                });
                const genreList = Array.from(allGenres);
                
                // Step 4: Create genre vectors for cosine similarity
                const likedGenres = new Set(likedMovie.genres);
                const likedVector = genreList.map(genre => likedGenres.has(genre) ? 1 : 0);
                
                const candidateMovies = movies.filter(movie => movie.id !== likedMovie.id);
                
                // Step 5: Calculate cosine similarity scores
                const scoredMovies = candidateMovies.map(candidate => {
                    const candidateGenres = new Set(candidate.genres);
                    const candidateVector = genreList.map(genre => candidateGenres.has(genre) ? 1 : 0);
                    
                    // Calculate dot product
                    let dotProduct = 0;
                    for (let i = 0; i < genreList.length; i++) {
                        dotProduct += likedVector[i] * candidateVector[i];
                    }
                    
                    // Calculate magnitudes
                    const likedMagnitude = Math.sqrt(likedVector.reduce((sum, val) => sum + val * val, 0));
                    const candidateMagnitude = Math.sqrt(candidateVector.reduce((sum, val) => sum + val * val, 0));
                    
                    // Calculate cosine similarity
                    const score = (likedMagnitude * candidateMagnitude) > 0 ? 
                        dotProduct / (likedMagnitude * candidateMagnitude) : 0;
                    
                    return {
                        ...candidate,
                        score: score
                    };
                });
                
                // Step 6: Sort by score in descending order
                scoredMovies.sort((a, b) => b.score - a.score);
                
                // Step 7: Select top recommendations
                const topRecommendations = scoredMovies.slice(0, 2);
                
                // Step 8: Display results
                if (topRecommendations.length > 0) {
                    const recommendationTitles = topRecommendations.map(movie => movie.title);
                    resultElement.textContent = `Because you liked "${likedMovie.title}", we recommend: ${recommendationTitles.join(', ')}`;
                    resultElement.className = 'success';
                } else {
                    resultElement.textContent = `No recommendations found for "${likedMovie.title}".`;
                    resultElement.className = 'error';
                }
            } catch (error) {
                console.error('Error in recommendation calculation:', error);
                resultElement.textContent = "An error occurred while calculating recommendations.";
                resultElement.className = 'error';
            }
        }, 100);
    } catch (error) {
        console.error('Error in getRecommendations:', error);
        resultElement.textContent = "An unexpected error occurred.";
        resultElement.className = 'error';
    }
}
