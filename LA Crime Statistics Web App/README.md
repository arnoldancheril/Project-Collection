# Los Angeles Crime Statistics Web Application

## Project Overview
This web application provides Los Angeles residents with comprehensive crime statistics to foster awareness and safety in their neighborhoods. By offering insights into local crime trends, the platform empowers individuals to make informed decisions regarding their security and contribute to a safer community.

## Features
- **Safety Score Calculator**: Calculate safety scores for specific areas in Los Angeles by entering a ZIP code
- **Filtered Data**: Explore crime statistics based on various filters:
  - Areas with highest crime rates
  - Time of day when crime is highest
  - Most prevalent crimes committed
  - Areas where specific crimes (like Grand Theft Auto or Battery) are highest
  - Demographic information about victims
- **Discussion Panel**: A space for community members to share observations, report incidents, and discuss safety concerns
- **About Page**: Information about the project's purpose, functionality, and technology stack

## Technology Stack
- **Frontend**: React, TypeScript, React Router
- **Backend**: Node.js, Express
- **Database**: MySQL
- **Cloud**: Google Cloud SQL

## Project Structure
```
.
├── backend/              # Node.js/Express server code
│   ├── index.js          # Main server file
│   └── package.json      # Backend dependencies
└── frontend/             # React/TypeScript client code
    ├── public/           # Static files
    ├── src/              # Source code
    │   ├── components/   # React components
    │   │   ├── About.tsx
    │   │   ├── Discussion.tsx
    │   │   ├── FilteredData.tsx
    │   │   ├── Home.tsx
    │   │   └── SafetyScore.tsx
    │   ├── images/       # Image assets
    │   ├── App.tsx       # Main application component
    │   └── index.tsx     # Application entry point
    └── package.json      # Frontend dependencies
```

## Purpose
This project was developed for a database class, focusing on creating a web application that utilizes SQL database queries to provide meaningful insights from crime data. The application serves as a practical tool for the Los Angeles community while demonstrating database integration with a modern web application architecture.