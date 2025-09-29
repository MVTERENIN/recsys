let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;

async function loadData() {
    try {
        console.log('Loading data from local files...');
        
        // Load movie data
        const moviesResponse = await fetch('./u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status} ${moviesResponse.statusText}`);
        }
        const moviesText = await moviesResponse.text();
        parseItemData(moviesText);

        // Load ratings data
        const ratingsResponse = await fetch('./u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load ratings data: ${ratingsResponse.status} ${ratingsResponse.statusText}`);
        }
        const ratingsText = await ratingsResponse.text();
        parseRatingData(ratingsText);
        
        console.log(`Data loaded successfully: ${movies.length} movies, ${ratings.length} ratings from ${numUsers} users`);
        return { movies, ratings, numUsers, numMovies };
    } catch (error) {
        console.error('Error loading data:', error);
        throw new Error(`Failed to load data: ${error.message}. Please ensure u.item and u.data files are in the same directory.`);
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
                // Only store id and title for the recommender
                movies.push({
                    id: movieId,
                    title: title
                });
            }
        }
    }
    
    numMovies = movies.length;
    console.log(`Parsed ${movies.length} movies`);
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
    console.log(`Parsed ${ratings.length} ratings from ${numUsers} users`);
}
