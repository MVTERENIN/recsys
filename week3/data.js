let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;

async function loadData() {
    try {
        // Use a CORS proxy to bypass restrictions
        const proxyUrl = 'https://cors-anywhere.herokuapp.com/';
        const moviesUrl = 'https://files.grouplens.org/datasets/movielens/ml-100k/u.item';
        const ratingsUrl = 'https://files.grouplens.org/datasets/movielens/ml-100k/u.data';
        
        // Load movie data through proxy
        const moviesResponse = await fetch(proxyUrl + moviesUrl);
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        const moviesText = await moviesResponse.text();
        parseItemData(moviesText);

        // Load ratings data through proxy
        const ratingsResponse = await fetch(proxyUrl + ratingsUrl);
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load ratings data: ${ratingsResponse.status}`);
        }
        const ratingsText = await ratingsResponse.text();
        parseRatingData(ratingsText);
        
        console.log(`Data loaded successfully: ${movies.length} movies, ${ratings.length} ratings`);
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
