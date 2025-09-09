// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Get all movies and populate dropdown
    const movies = getAllMovies();
    populateMovieDropdown(movies);
    
    // Set up event listener for the button
    document.getElementById('recommend-btn').addEventListener('click', getRecommendation);
});

// Populate the movie dropdown
function populateMovieDropdown(movies) {
    const selectElement = document.getElementById('movie-select');
    
    // Clear existing options except the first one
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }
    
    // Add movies to dropdown
    movies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        selectElement.appendChild(option);
    });
}

// Get and display recommendation
function getRecommendation() {
    const selectElement = document.getElementById('movie-select');
    const movieId = parseInt(selectElement.value);
    
    if (!movieId) {
        alert("Please select a movie first!");
        return;
    }
    
    const result = getRecommendation(movieId);
    
    if (!result) {
        document.getElementById('result').textContent = "No recommendation found for this movie.";
        return;
    }
    
    displayRecommendation(movieId, result.movie, result.similarity);
}

// Display the recommendation
function displayRecommendation(originalId, recommendedMovie, similarity) {
    const originalMovie = getAllMovies().find(movie => movie.id === originalId);
    
    const resultElement = document.getElementById('result');
    resultElement.innerHTML = `
        <div class="movie-pair">
            <div class="movie-card original-movie">
                <div class="movie-title">${originalMovie.title}</div>
                <div class="movie-genres">
                    ${originalMovie.genres.map(genre => `<span class="genre-tag">${genre}</span>`).join('')}
                </div>
            </div>
            
            <div class="connection">⇨</div>
            
            <div class="movie-card recommended-movie">
                <div class="movie-title">${recommendedMovie.title}</div>
                <div class="movie-genres">
                    ${recommendedMovie.genres.map(genre => `<span class="genre-tag">${genre}</span>`).join('')}
                </div>
            </div>
        </div>
        <div class="similarity">Similarity: ${similarity}</div>
    `;
}
