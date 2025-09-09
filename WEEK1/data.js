// Predefined movie pairs - each movie with its closest match
const moviePairs = [
    {
        original: { id: 1, title: "The Shawshank Redemption", genres: ["Drama"] },
        recommended: { id: 2, title: "The Green Mile", genres: ["Crime", "Drama", "Fantasy"] },
        similarity: "92%"
    },
    {
        original: { id: 3, title: "The Dark Knight", genres: ["Action", "Crime", "Drama"] },
        recommended: { id: 4, title: "Batman Begins", genres: ["Action", "Adventure"] },
        similarity: "89%"
    },
    {
        original: { id: 5, title: "Pulp Fiction", genres: ["Crime", "Drama"] },
        recommended: { id: 6, title: "Reservoir Dogs", genres: ["Crime", "Drama", "Thriller"] },
        similarity: "87%"
    },
    {
        original: { id: 7, title: "Forrest Gump", genres: ["Drama", "Romance"] },
        recommended: { id: 8, title: "The Curious Case of Benjamin Button", genres: ["Drama", "Fantasy", "Romance"] },
        similarity: "85%"
    },
    {
        original: { id: 9, title: "Inception", genres: ["Action", "Adventure", "Sci-Fi"] },
        recommended: { id: 10, title: "The Matrix", genres: ["Action", "Sci-Fi"] },
        similarity: "90%"
    },
    {
        original: { id: 11, title: "The Lord of the Rings: The Fellowship of the Ring", genres: ["Adventure", "Drama", "Fantasy"] },
        recommended: { id: 12, title: "The Hobbit: An Unexpected Journey", genres: ["Adventure", "Fantasy"] },
        similarity: "88%"
    },
    {
        original: { id: 13, title: "Goodfellas", genres: ["Biography", "Crime", "Drama"] },
        recommended: { id: 14, title: "The Godfather", genres: ["Crime", "Drama"] },
        similarity: "91%"
    },
    {
        original: { id: 15, title: "The Avengers", genres: ["Action", "Adventure", "Sci-Fi"] },
        recommended: { id: 16, title: "Guardians of the Galaxy", genres: ["Action", "Adventure", "Comedy"] },
        similarity: "84%"
    }
];

// Get all original movies for the dropdown
function getAllMovies() {
    return moviePairs.map(pair => pair.original);
}

// Find recommendation for a movie ID
function getRecommendation(movieId) {
    const pair = moviePairs.find(pair => pair.original.id === movieId);
    return pair ? { movie: pair.recommended, similarity: pair.similarity } : null;
}
