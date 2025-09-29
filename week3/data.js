let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;

async function loadData() {
    try {
        // Load movie data
        const moviesResponse = await fetch('https://files.grouplens.org/datasets/movielens/ml-100k/u.item');
        const moviesText = await moviesResponse.text();
        parseItemData(moviesText);

        // Load ratings data
        const ratingsResponse = await fetch('https://files.grouplens.org/datasets/movielens/ml-100k/u.data');
        const ratingsText = await ratingsResponse.text();
        parseRatingData(ratingsText);
        
        return { movies, ratings, numUsers, numMovies };
    } catch (error) {
        console.error('Error loading data:', error);
        throw error;
    }
}

function parseItemData(text) {
    movies = [];
    const lines = text.split('\n');
    
    for (const line of lines) {
        if (line.trim()) {
            const parts = line.split('|');
            if (parts.length >= 2) {
                const movieId = parseInt(parts[0]);
                const title = parts[1];
                movies.push({
                    id: movieId,
                    title: title
                });
            }
        }
    }
    
    numMovies = movies.length;
    console.log(`Loaded ${movies.length} movies`);
}

function parseRatingData(text) {
    ratings = [];
    const userSet = new Set();
    const lines = text.split('\n');
    
    for (const line of lines) {
        if (line.trim()) {
            const parts = line.split('\t');
            if (parts.length >= 3) {
                const userId = parseInt(parts[0]);
                const movieId = parseInt(parts[1]);
                const rating = parseFloat(parts[2]);
                
                ratings.push({
                    userId: userId,
                    movieId: movieId,
                    rating: rating
                });
                
                userSet.add(userId);
            }
        }
    }
    
    numUsers = userSet.size;
    console.log(`Loaded ${ratings.length} ratings from ${numUsers} users`);
}
