# Project Collection
A comprehensive collection of my software development projects spanning various technologies, frameworks, and domains.

## Projects

### AI Article Summarizer
A web application built with Flask that provides intelligent article summarization capabilities using both traditional NLP techniques and AI-based transformers.

- **Features:**
  - Dual summarization methods: NLP-based (NLTK) and AI-based (HuggingFace transformers)
  - Text input or file upload options
  - Clean, responsive interface
  - Customizable summary length
  - Side-by-side comparison of different summarization techniques
- **Technologies:** Flask, PyTorch/TensorFlow, NLTK, HuggingFace Transformers, HTML/CSS, JavaScript
- **Use Case:** Content analysis, research, information extraction, document processing
- **Implementation Details:**
  - Extractive summarization using TextRank algorithm via NLTK
  - Abstractive summarization using pre-trained transformer models
  - RESTful API design for easy integration with other services
  - Responsive frontend for mobile and desktop use

### AlgoNest
A subscription-based platform for algorithmic trading bots with various risk profiles, featuring comprehensive performance visualization and monitoring tools.

- **Features:**
  - User authentication system
  - Bot listing with detailed descriptions and risk levels
  - Interactive performance dashboards with Chart.js visualizations
  - Bot algorithm transparency with syntax-highlighted code
  - Subscription and pricing management
  - Historical performance tracking
  - Risk assessment metrics
- **Technologies:** Django, Bootstrap 5, JavaScript, Chart.js, Prism.js, SQLite, RESTful APIs
- **Use Case:** Fintech, algorithmic trading, investment management, portfolio analysis
- **Implementation Details:**
  - MVC architecture using Django's framework
  - Object-relational mapping for database operations
  - JWT-based authentication
  - Responsive UI with interactive data visualizations
  - Integration with financial data APIs

### Database Management System
A custom-built relational database management system (DBMS) implementing core database functionality from scratch.

- **Features:**
  - SQL parsing and execution
  - ACID-compliant transactions
  - Concurrency control with locking mechanisms
  - B-tree indexing for optimized queries
  - Data type validation
  - Query optimization
  - Database recovery mechanisms
- **Technologies:** Python, SQLite
- **Use Case:** Data storage, query processing, transaction management, database administration
- **Implementation Details:**
  - Parser for SQL syntax using grammar rules
  - Query execution engine with support for joins, aggregations, and subqueries
  - Transaction manager with two-phase locking protocol
  - B-tree implementation for efficient indexing
  - Buffer manager for caching frequently accessed pages
  - Recovery system using write-ahead logging

### Deep Learning Loinc Standardization
A machine learning solution for standardizing laboratory test codes according to LOINC (Logical Observation Identifiers Names and Codes) standards in healthcare data.

- **Features:**
  - Two-stage fine-tuning approach with triplet loss
  - Data augmentation for improved generalization
  - Threshold-based classification for handling unmappable cases
  - Extensive evaluation framework with error analysis
  - Confidence scoring for matches
  - Interactive demo for manual verification
- **Technologies:** PyTorch, TensorFlow, Sentence Transformers, scikit-learn, Pandas
- **Use Case:** Healthcare data interoperability, medical informatics, standardization, clinical data analysis
- **Implementation Details:**
  - Pre-trained BERT models fine-tuned on medical terminology
  - Triplet loss function to learn embeddings for lab test descriptions
  - Custom data preprocessing pipeline for medical text
  - Transfer learning approach for limited labeled data
  - K-nearest neighbors algorithm for finding similar LOINC codes

### Havyn
A roommate matching platform that helps users find compatible living partners based on lifestyle preferences and habits.

- **Features:**
  - Profile creation and management
  - Compatibility-based matching algorithm
  - Swipe interface for easy interaction
  - In-app messaging system
  - Location preferences and filtering
  - Verification system for safety
- **Technologies:** React Native, Node.js, Express, MongoDB, Firebase, Google Maps API
- **Use Case:** Housing, roommate matching, lifestyle compatibility, rental management
- **Implementation Details:**
  - MERN stack architecture (MongoDB, Express, React Native, Node.js)
  - RESTful API design for backend services
  - Real-time messaging using Socket.io
  - Geolocation services for area-based matching
  - Weighted compatibility algorithm based on user preferences
  - Push notifications for new matches and messages

### School Scheduler
A comprehensive scheduling application to help students manage assignments, classes, and academic events.

- **Features:**
  - Task management with priorities and due dates
  - Calendar-based event scheduling
  - Authentication system
  - Analytics dashboard
  - Notification system for upcoming deadlines
  - Grade tracking
  - Course management
  - Study time recommendations
- **Technologies:** Python, Tkinter, SQLite, Matplotlib, bcrypt, pandas
- **Use Case:** Student productivity, academic planning, time management, educational analytics
- **Implementation Details:**
  - MVC architecture for clean separation of concerns
  - Local database with SQL queries for data persistence
  - Custom calendar widget with zoom functionality
  - Priority queue implementation for task sorting
  - Password hashing for secure authentication
  - Data visualization of study patterns and grade trends

### TicTacToeAI
An implementation of the classic Tic-Tac-Toe game with an unbeatable AI opponent using the minimax algorithm.

- **Features:**
  - Clean graphical interface
  - Player vs. AI gameplay
  - Intelligent AI using minimax algorithm with alpha-beta pruning
  - Score tracking
  - Symbol selection (X or O)
  - Multiple difficulty levels
  - Game history tracking
- **Technologies:** Python, Tkinter
- **Use Case:** Game development, AI algorithm demonstration, educational, recreational
- **Implementation Details:**
  - Minimax algorithm with a depth parameter for variable difficulty
  - Alpha-beta pruning for performance optimization
  - Object-oriented design with Game, Board, and Player classes
  - Event-driven programming for UI interactions
  - State management for game progression

### LA Crime Statistics Web App
A web application that provides comprehensive crime statistics for Los Angeles neighborhoods to foster awareness and community safety.

- **Features:**
  - Safety Score Calculator for specific areas by ZIP code
  - Filtered crime data visualization (highest crime rates, time patterns)
  - Crime type analysis (e.g., Grand Theft Auto, Battery)
  - Demographic information about victims
  - Community discussion panel for sharing observations
  - About page with project information
- **Technologies:** React, TypeScript, Node.js, Express, MySQL, Google Cloud SQL
- **Use Case:** Community safety, urban planning, law enforcement resource allocation, real estate decisions
- **Implementation Details:**
  - React frontend with TypeScript for type safety
  - RESTful API design with Node.js/Express backend
  - SQL database queries for statistical analysis
  - Responsive design for mobile and desktop usage
  - Component-based architecture for maintainability

### Flight Route Analysis Tool
A data analysis tool that uses graph algorithms to analyze airport connections and find optimal flight routes using real-world flight data.

- **Features:**
  - Breadth-First Search (BFS) for network traversal
  - Dijkstra's Algorithm for shortest path finding
  - PageRank Algorithm for airport importance ranking
  - Visualization of flight paths
  - Command-line interface for various operations
  - Comprehensive test suite
- **Technologies:** C++, Python (for data processing), Catch2 (testing framework)
- **Use Case:** Travel planning, airline route optimization, network analysis, transportation logistics
- **Implementation Details:**
  - Graph representation using adjacency lists
  - Geographic distance calculation between airports
  - Custom data parsers for OpenFlights.org datasets
  - Algorithm implementations optimized for large networks
  - Modular design with separation of data structures and algorithms
  - Thorough testing with over 6 test cases for different scenarios
