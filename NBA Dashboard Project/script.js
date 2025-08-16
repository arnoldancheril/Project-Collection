const CONFIG = {

    width: 1000,
    height: 600,
    margin: { top: 60, right: 80, bottom: 80, left: 80 },
    

    transitionDuration: 1000,
    delayBetweenElements: 50,
    

    colors: {
        threePt: '#ea580c',
        midRange: '#1e40af', 
        efficiency: '#16a34a',
        accent: '#f59e0b',
        dark: '#0f172a',
        gray: '#64748b',
        background: '#f8fafc'
    }
};


let state = {
    currentScene: 0,
    totalScenes: 5,
    data: {},
    analysisType: 'players', // 'players' or 'teams'
    selectedPlayer: 'Stephen Curry',
    selectedPlayers: ['Stephen Curry'],
    selectedTeams: [],
    startYear: 2004,
    endYear: 2024,
    viewMode: 'evolution',
    filters: {
        playoffOnly: false,
        winningTeams: false
    },
    isPlaying: false,
    animationFrame: null,
    availablePlayers: [],
    availableTeams: []
};

// Scene configurations
const scenes = [
    {
        title: "The Great Transformation",
        subtitle: "From mid-range shooting to three-point shots - see how basketball fundamentally changed",
        content: "In 2004, NBA teams took just 18.7% of their shots from beyond the arc. By 2024, that number skyrocketed to 39.5% - more than doubling in just two decades."
    },
    {
        title: "Evolution Timeline", 
        subtitle: "Watch the three-point revolution unfold year by year across the entire league",
        content: "The transformation wasn't overnight - it was a steady evolution driven by analytics, strategy, and revolutionary players who changed the game forever."
    },
    {
        title: "Revolutionary Players",
        subtitle: "Steph Curry, the sharpshooter, who led the charge and redefined basketball",
        content: "Stephen Curry's record-breaking 402 made threes in 2015-16 wasn't just a record - it was a statement that changed how every team approaches offense."
    },
    {
        title: "The Efficiency Revolution", 
        subtitle: "How three-point shooting became increasingly effective over time",
        content: "As teams shifted from mid-range to three-point shots, league-wide scoring efficiency reached historic highs. The math is simple: 3 > 2."
    },
    {
        title: "Interactive Explorer",
        subtitle: "Dive deep into the data - explore players, compare eras, and discover insights",
        content: "Now it's your turn to explore. Select players, adjust time ranges, and discover the stories hidden in 4.2 million shots."
    }
];

// Utility functions
const utils = {
    // Responsive SVG setup
    setupSVG: function(selector, width = CONFIG.width, height = CONFIG.height) {
        const container = d3.select(selector);
        
        // Validate container exists
        if (container.empty()) {
            throw new Error(`SVG container element not found: ${selector}`);
        }
        
        // Clear existing content
        container.selectAll("*").remove();
        
        // Get container dimensions safely
        const containerNode = container.node();
        if (!containerNode) {
            throw new Error(`Container node is null: ${selector}`);
        }
        
        let containerWidth;
        try {
            const containerRect = containerNode.getBoundingClientRect();
            containerWidth = containerRect.width;
            
            // Fallback if container has no dimensions
            if (containerWidth === 0) {
                console.warn(`Container ${selector} has zero width, using parent or default`);
                const parent = containerNode.parentElement;
                containerWidth = parent ? parent.getBoundingClientRect().width : 1000;
            }
        } catch (error) {
            console.warn(`Could not get container dimensions for ${selector}, using default:`, error);
            containerWidth = 1000;
        }
        
        const aspectRatio = width / height;
        
        // Responsive width calculation with minimum size
        const responsiveWidth = Math.min(width, Math.max(containerWidth - 40, 300));
        const responsiveHeight = responsiveWidth / aspectRatio;
        
        const svg = container
            .attr('width', responsiveWidth)
            .attr('height', responsiveHeight)
            .attr('viewBox', `0 0 ${width} ${height}`)
            .attr('preserveAspectRatio', 'xMidYMid meet');
            
        const g = svg.append('g')
            .attr('transform', `translate(${CONFIG.margin.left}, ${CONFIG.margin.top})`);
            
        return { svg, g, width: width - CONFIG.margin.left - CONFIG.margin.right, 
                height: height - CONFIG.margin.top - CONFIG.margin.bottom };
    },
    
    // Smart axis formatting to prevent overcrowding
    getTimeAxis: function(xScale, width) {
        const domain = xScale.domain();
        const range = domain[1] - domain[0];
        
        let tickInterval;
        if (range <= 5) tickInterval = 1;
        else if (range <= 10) tickInterval = 2;
        else if (range <= 20) tickInterval = 4;
        else tickInterval = 5;
        
        const ticks = [];
        for (let year = Math.ceil(domain[0] / tickInterval) * tickInterval; year <= domain[1]; year += tickInterval) {
            ticks.push(year);
        }
        
        return d3.axisBottom(xScale)
            .tickValues(ticks)
            .tickFormat(d3.format('d'));
    },
    
    // Enhanced tooltip
    showTooltip: function(event, content) {
        const tooltip = d3.select('#tooltip');
        tooltip.html(content)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .classed('show', true);
    },
    
    hideTooltip: function() {
        d3.select('#tooltip').classed('show', false);
    },
    
    // Format large numbers
    formatNumber: function(num) {
        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
        if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
        return num.toString();
    }
};

// Initialize application
document.addEventListener('DOMContentLoaded', async function() {
    showLoading();
    
    try {
        console.log('🏀 Starting NBA Dashboard initialization...');
        
        await loadAllData();
        console.log('✅ Data loaded successfully');
        
        initializeEventListeners();
        console.log('✅ Event listeners initialized');
        
        updateScene();
        console.log('✅ Initial scene rendered');
        
        hideLoading();
        console.log('🎉 NBA Dashboard fully loaded!');
        
    } catch (error) {
        console.error('❌ Failed to load application:', error);
        showError(`Failed to load NBA data: ${error.message}<br><br>Please check your internet connection and refresh the page.`);
    }
});

// Data loading
async function loadAllData() {
    try {
        const [scene1, scene2, scene3, scene4] = await Promise.all([
            d3.json('data/scene1_data.json'),
            d3.json('data/scene2_data.json'), 
            d3.json('data/scene3_data.json'),
            d3.json('data/scene4_data.json')
        ]);
        
        state.data = { scene1, scene2, scene3, scene4 };
        
        // Initialize available players and teams
        initializeSearchData();
        
        // Enhance data with additional calculations
        enhanceDataWithCalculations();
        
    } catch (error) {
        throw new Error('Failed to load data files: ' + error.message);
    }
}

function initializeSearchData() {
    // Extract all available players from comprehensive data
    const playersFromScene3 = state.data.scene3.map(p => p.player);
    const comprehensivePlayers = state.data.scene4.player_data ? 
        state.data.scene4.player_data.map(p => p.player) : [];
    
    state.availablePlayers = [...new Set([...playersFromScene3, ...comprehensivePlayers])].sort();
    
    // Extract teams from comprehensive data
    state.availableTeams = state.data.scene4.team_data ? 
        state.data.scene4.team_data.map(t => t.team).sort() : [];
    
    // Initialize conference-based team interface
    initializeConferenceTeamSelection();
    
    console.log(`🏀 Initialized with ${state.availablePlayers.length} players and ${state.availableTeams.length} teams`);
}

function initializeConferenceTeamSelection() {
    if (!state.data.scene4.team_conferences) return;
    
    const conferences = state.data.scene4.team_conferences;
    
    // Populate Eastern Conference
    const easternDivisions = document.getElementById('eastern-divisions');
    populateConference(easternDivisions, conferences['Eastern Conference']);
    
    // Populate Western Conference
    const westernDivisions = document.getElementById('western-divisions');
    populateConference(westernDivisions, conferences['Western Conference']);
    
    // Add conference tab functionality
    document.querySelectorAll('.conference-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            document.querySelectorAll('.conference-tab').forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
            
            const conference = e.target.dataset.conference;
            document.getElementById('eastern-divisions').style.display = 
                conference === 'Eastern Conference' ? 'grid' : 'none';
            document.getElementById('western-divisions').style.display = 
                conference === 'Western Conference' ? 'grid' : 'none';
        });
    });
}

function populateConference(container, divisions) {
    Object.entries(divisions).forEach(([divisionName, teams]) => {
        const divisionContainer = container.querySelector(`[data-division="${divisionName}"] .team-list`);
        if (divisionContainer) {
            divisionContainer.innerHTML = '';
            teams.forEach(team => {
                const teamOption = document.createElement('div');
                teamOption.className = 'team-option';
                teamOption.textContent = team.team;
                teamOption.dataset.team = team.team;
                teamOption.addEventListener('click', () => selectTeam(team.team));
                divisionContainer.appendChild(teamOption);
            });
        }
    });
}

function selectTeam(teamName) {
    if (!state.selectedTeams.includes(teamName)) {
        state.selectedTeams.push(teamName);
        updateSelectedItems('teams');
        
        // Update visual selection
        document.querySelectorAll('.team-option').forEach(option => {
            if (option.dataset.team === teamName) {
                option.classList.add('selected');
            }
        });
        
        if (state.currentScene === 4) updateExplorerVisualization();
    }
}

function enhanceDataWithCalculations() {
    try {
        console.log('🔄 Enhancing data with calculations...');
        
        // Add trend calculations, player rankings, etc.
        const leagueData = state.data.scene2;
        
        if (!leagueData || !Array.isArray(leagueData)) {
            throw new Error('Scene2 data is missing or invalid');
        }
        
        // Calculate year-over-year changes
        leagueData.forEach((d, i) => {
            if (i > 0) {
                d.threePtChange = d.three_pt_rate - leagueData[i-1].three_pt_rate;
                d.efficiencyChange = d.efg_percentage - leagueData[i-1].efg_percentage;
            } else {
                d.threePtChange = 0;
                d.efficiencyChange = 0;
            }
        });
        
        // Enhance player data
        if (state.data.scene3 && Array.isArray(state.data.scene3)) {
            state.data.scene3.forEach(player => {
                if (player.seasons && Array.isArray(player.seasons)) {
                    player.seasons.forEach((season, i) => {
                        if (i > 0) {
                            season.careerProgression = season.three_pt_rate - player.seasons[0].three_pt_rate;
                        } else {
                            season.careerProgression = 0;
                        }
                    });
                }
            });
        }
        
        console.log('✅ Data enhancement complete');
        
    } catch (error) {
        console.error('❌ Error enhancing data:', error);
        throw error;
    }
}

// Event listeners
function initializeEventListeners() {
    // Scene navigation
    document.querySelectorAll('.nav-pill').forEach(pill => {
        pill.addEventListener('click', (e) => {
            const scene = parseInt(e.target.dataset.scene);
            changeScene(scene);
        });
    });
    
    // Progress dots
    document.querySelectorAll('.dot').forEach(dot => {
        dot.addEventListener('click', (e) => {
            const scene = parseInt(e.target.dataset.scene);
            changeScene(scene);
        });
    });
    
    // Navigation buttons - safe access
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    if (prevBtn) prevBtn.addEventListener('click', () => changeScene(state.currentScene - 1));
    if (nextBtn) nextBtn.addEventListener('click', () => changeScene(state.currentScene + 1));
    
    // Scene navigation buttons (smaller ones under tabs) - safe access
    const scenePrevBtn = document.getElementById('scene-prev-btn');
    const sceneNextBtn = document.getElementById('scene-next-btn');
    if (scenePrevBtn) scenePrevBtn.addEventListener('click', () => changeScene(state.currentScene - 1));
    if (sceneNextBtn) sceneNextBtn.addEventListener('click', () => changeScene(state.currentScene + 1));
    
    // Analysis type toggle
    document.querySelectorAll('.analysis-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.analysis-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            state.analysisType = e.target.dataset.type;
            toggleAnalysisControls();
            
            // Update enhanced explorer if it's loaded
            if (window.nbaAdvancedExplorer && state.currentScene === 4) {
                const viewType = e.target.dataset.type; // 'players' or 'teams'
                window.nbaAdvancedExplorer.setCurrentView(viewType);
            } else if (state.currentScene === 4) {
                updateExplorerVisualization();
            }
        });
    });

    // Search functionality
    initializeSearchControls();
    
    // Year sliders - safe access
    const startYear = document.getElementById('start-year');
    const endYear = document.getElementById('end-year');
    if (startYear) {
        startYear.addEventListener('input', (e) => {
            state.startYear = parseInt(e.target.value);
            const label = document.getElementById('start-year-label');
            if (label) label.textContent = state.startYear;
            if (state.currentScene === 4) updateExplorerVisualization();
        });
    }
    if (endYear) {
        endYear.addEventListener('input', (e) => {
            state.endYear = parseInt(e.target.value);
            const label = document.getElementById('end-year-label');
            if (label) label.textContent = state.endYear;
            if (state.currentScene === 4) updateExplorerVisualization();
        });
    }
    
    // View toggles
    document.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            state.viewMode = e.target.dataset.view;
            updateChartTitle();
            if (state.currentScene === 4) updateExplorerVisualization();
        });
    });

    // Filter controls - safe access
    const playoffOnly = document.getElementById('playoff-only');
    const winningTeams = document.getElementById('winning-teams');
    if (playoffOnly) {
        playoffOnly.addEventListener('change', (e) => {
            state.filters.playoffOnly = e.target.checked;
            if (state.currentScene === 4) updateExplorerVisualization();
        });
    }
    if (winningTeams) {
        winningTeams.addEventListener('change', (e) => {
            state.filters.winningTeams = e.target.checked;
            if (state.currentScene === 4) updateExplorerVisualization();
        });
    }
    
    // Timeline play button - safe access
    const playBtn = document.getElementById('play-btn');
    if (playBtn) playBtn.addEventListener('click', toggleTimelineAnimation);
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft' && state.currentScene > 0) {
            changeScene(state.currentScene - 1);
        } else if (e.key === 'ArrowRight' && state.currentScene < state.totalScenes - 1) {
            changeScene(state.currentScene + 1);
        }
    });
}

// Scene management
function changeScene(newScene) {
    if (newScene < 0 || newScene >= state.totalScenes || newScene === state.currentScene) return;
    
    state.currentScene = newScene;
    updateScene();
}

function updateScene() {
    // Update navigation
    updateNavigation();
    
    // Update scene content
    updateSceneContent();
    
    // Hide/show appropriate elements
    updateSceneVisibility();
    
    // Draw scene visualization
    drawCurrentScene();
}

function updateNavigation() {
    // Update nav pills
    document.querySelectorAll('.nav-pill').forEach((pill, index) => {
        pill.classList.toggle('active', index === state.currentScene);
    });
    
    // Update progress dots
    document.querySelectorAll('.dot').forEach((dot, index) => {
        dot.classList.toggle('active', index === state.currentScene);
    });
    
    // Update navigation buttons (both bottom and scene nav buttons)
    document.getElementById('prev-btn').disabled = state.currentScene === 0;
    document.getElementById('next-btn').disabled = state.currentScene === state.totalScenes - 1;
    
    // Update scene navigation buttons
    const scenePrevBtn = document.getElementById('scene-prev-btn');
    const sceneNextBtn = document.getElementById('scene-next-btn');
    if (scenePrevBtn) scenePrevBtn.disabled = state.currentScene === 0;
    if (sceneNextBtn) sceneNextBtn.disabled = state.currentScene === state.totalScenes - 1;
    
    // Update scene counter
    document.getElementById('current-scene-num').textContent = state.currentScene + 1;
}

function updateSceneContent() {
    const scene = scenes[state.currentScene];
    
    document.getElementById('scene-title').textContent = scene.title;
    document.getElementById('scene-subtitle').textContent = scene.subtitle;
    
    // Add fade-in animation
    document.querySelector('.scene-description').classList.add('fade-in');
    setTimeout(() => {
        document.querySelector('.scene-description').classList.remove('fade-in');
    }, 500);
}

function updateSceneVisibility() {
    // Show/hide explorer dashboard
    const explorerDashboard = document.getElementById('explorer-dashboard');
    const timelineNav = document.getElementById('timeline-nav');
    const secondaryPanels = document.getElementById('secondary-panels');
    const mainVisualization = document.getElementById('main-visualization');
    
    explorerDashboard.style.display = state.currentScene === 4 ? 'block' : 'none';
    timelineNav.style.display = state.currentScene === 1 ? 'block' : 'none';
    
    // Show/hide main visualization - hide for explorer, show for all other scenes
    mainVisualization.style.display = state.currentScene === 4 ? 'none' : 'block';
    
    // Clear secondary panels
    secondaryPanels.innerHTML = '';
}

function drawCurrentScene() {
    switch (state.currentScene) {
        case 0: drawScene0_Overview(); break;
        case 1: drawScene1_Evolution(); break; 
        case 2: drawScene2_Players(); break;
        case 3: drawScene3_Efficiency(); break;
        case 4: drawScene4_Explorer(); break;
    }
}

// Scene 0: Enhanced Overview with better design
function drawScene0_Overview() {
    try {

        const { svg, g, width, height } = utils.setupSVG('#main-visualization');

        
        const data = state.data.scene1;
        if (!data) {
            throw new Error('Scene1 data is missing');
        }
        console.log('✅ Scene1 data loaded:', data);
    
    // Create enhanced comparison data
    const comparisonData = [
        { 
            year: '2004', 
            threePt: data['2004'].three_pt_percentage, 
            midRange: data['2004'].mid_range_percentage,
            total: data['2004'].total_shots
        },
        { 
            year: '2024', 
            threePt: data['2024'].three_pt_percentage, 
            midRange: data['2024'].mid_range_percentage,
            total: data['2024'].total_shots
        }
    ];
    
    // Enhanced scales with better spacing
    const xScale = d3.scaleBand()
        .domain(['2004', '2024'])
        .range([0, width])
        .padding(0.4);
    
    const yScale = d3.scaleLinear()
        .domain([0, 50])
        .range([height, 0]);
    
    // Add title with better positioning
    g.append('text')
        .attr('class', 'chart-title')
        .attr('x', width / 2)
        .attr('y', -30)
        .style('font-size', '24px')
        .style('font-weight', '700')
        .text('The Three-Point Transformation');
    
    // Create grouped bars with better spacing
    const barGroups = g.selectAll('.bar-group')
        .data(comparisonData)
        .enter()
        .append('g')
        .attr('class', 'bar-group')
        .attr('transform', d => `translate(${xScale(d.year)}, 0)`);
    
    const barWidth = xScale.bandwidth() / 2.5;
    const barSpacing = 8;
    
    // Three-point bars with animation
    barGroups.append('rect')
        .attr('class', 'bar-three-pt')
        .attr('x', 0)
        .attr('y', height)
        .attr('width', barWidth)
        .attr('height', 0)
        .attr('rx', 4)
        .transition()
        .duration(CONFIG.transitionDuration)
        .delay((d, i) => i * 200)
        .attr('y', d => yScale(d.threePt))
        .attr('height', d => height - yScale(d.threePt));
    
    // Mid-range bars with animation
    barGroups.append('rect')
        .attr('class', 'bar-mid-range')
        .attr('x', barWidth + barSpacing)
        .attr('y', height)
        .attr('width', barWidth)
        .attr('height', 0)
        .attr('rx', 4)
        .transition()
        .duration(CONFIG.transitionDuration)
        .delay((d, i) => i * 200 + 100)
        .attr('y', d => yScale(d.midRange))
        .attr('height', d => height - yScale(d.midRange));
    
    // Enhanced Y-axis with proper spacing
    const yAxis = g.append('g')
        .attr('class', 'axis')
        .style('opacity', 0)
        .call(d3.axisLeft(yScale)
            .tickFormat(d => d + '%')
            .tickSize(-width)
            .ticks(6))
        .transition()
        .duration(CONFIG.transitionDuration)
        .style('opacity', 1);
    
    // Style the grid lines
    yAxis.selectAll('.tick line')
        .attr('stroke', CONFIG.colors.gray)
        .attr('stroke-dasharray', '2,2')
        .attr('opacity', 0.3);
    
    // Add year labels under bars
    g.append('text')
        .attr('x', xScale('2004') + xScale.bandwidth() / 2)
        .attr('y', height + 40)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('font-weight', '600')
        .style('fill', CONFIG.colors.dark)
        .text('2003-2004');
        
    g.append('text')
        .attr('x', xScale('2024') + xScale.bandwidth() / 2)
        .attr('y', height + 40)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('font-weight', '600')
        .style('fill', CONFIG.colors.dark)
        .text('2023-2024');

    // Add axis labels
    g.append('text')
        .attr('class', 'axis-label')
        .attr('transform', 'rotate(-90)')
        .attr('y', -50)
        .attr('x', -height / 2)
        .attr('text-anchor', 'middle')
        .style('font-size', '16px')
        .text('Percentage of Total Shots');
    
    // Enhanced legend
    const legend = g.append('g')
        .attr('transform', `translate(${width - 200}, 30)`)
        .style('opacity', 0);
    
    const legendData = [
        { label: '3-Point Shots', color: CONFIG.colors.threePt },
        { label: 'Mid-Range Shots', color: CONFIG.colors.midRange }
    ];
    
    const legendItems = legend.selectAll('.legend-item')
        .data(legendData)
        .enter()
        .append('g')
        .attr('class', 'legend-item')
        .attr('transform', (d, i) => `translate(0, ${i * 25})`);
    
    legendItems.append('rect')
        .attr('width', 18)
        .attr('height', 18)
        .attr('rx', 3)
        .attr('fill', d => d.color);
    
    legendItems.append('text')
        .attr('x', 25)
        .attr('y', 13)
        .style('font-size', '14px')
        .style('font-weight', '500')
        .style('fill', CONFIG.colors.dark)
        .text(d => d.label);
    
    legend.transition()
        .duration(CONFIG.transitionDuration)
        .delay(1000)
        .style('opacity', 1);
    
    // Add secondary panels with insights
    createSecondaryPanels([
        {
            title: "📈 Dramatic Shift",
            content: `In just 20 years, three-point shooting more than doubled from ${data['2004'].three_pt_percentage}% to ${data['2024'].three_pt_percentage}% of all shots.`
        },
        {
            title: "📉 Mid-Range Decline", 
            content: `Mid-range shots dropped dramatically from ${data['2004'].mid_range_percentage}% to ${data['2024'].mid_range_percentage}% - a 68% decrease.`
        },
        {
            title: "🏀 Total Volume",
            content: `${utils.formatNumber(data['2024'].total_shots)} total shots analyzed in 2024 season vs ${utils.formatNumber(data['2004'].total_shots)} in 2004.`
        }
    ]);
    
    console.log('✅ Scene 0 completed successfully');
    
    } catch (error) {
        console.error('❌ Error in drawScene0_Overview:', error);
        showError(`Failed to render overview scene: ${error.message}`);
        throw error;
    }
}

// Scene 1: Enhanced Evolution Timeline with fixed x-axis
function drawScene1_Evolution() {
    const { svg, g, width, height } = utils.setupSVG('#main-visualization');
    const data = state.data.scene2;
    
    // Enhanced scales
    const xScale = d3.scaleLinear()
        .domain(d3.extent(data, d => d.season))
        .range([0, width]);
    
    const yScale = d3.scaleLinear()
        .domain([0, 45])
        .range([height, 0]);
    
    // Add background grid with proper x-axis spacing
    const xAxis = g.append('g')
        .attr('class', 'axis')
        .attr('transform', `translate(0, ${height})`)
        .call(utils.getTimeAxis(xScale, width));
    
    const yAxis = g.append('g')
        .attr('class', 'axis')
        .call(d3.axisLeft(yScale).tickFormat(d => d + '%').tickSize(-width));
    
    // Style grid
    yAxis.selectAll('.tick line')
        .attr('stroke', CONFIG.colors.gray)
        .attr('stroke-dasharray', '2,2')
        .attr('opacity', 0.3);
    
    xAxis.selectAll('.tick line')
        .attr('stroke', CONFIG.colors.gray)
        .attr('stroke-dasharray', '2,2')
        .attr('opacity', 0.3);
    
    // Enhanced line generators
    const threePtLine = d3.line()
        .x(d => xScale(d.season))
        .y(d => yScale(d.three_pt_rate))
        .curve(d3.curveMonotoneX);
    
    const midRangeLine = d3.line()
        .x(d => xScale(d.season))
        .y(d => yScale(d.mid_range_rate))
        .curve(d3.curveMonotoneX);
    
    // Draw lines with proper stroke width
    const threePtPath = g.append('path')
        .datum(data)
        .attr('class', 'three-pt-line')
        .attr('d', threePtLine)
        .style('stroke-width', 4);
    
    const midRangePath = g.append('path')
        .datum(data)
        .attr('class', 'mid-range-line')
        .attr('d', midRangeLine)
        .style('stroke-width', 4);
    
    // Add interactive points with better hover effects
    const pointsGroup = g.append('g').attr('class', 'points-group');
    
    pointsGroup.selectAll('.three-pt-point')
        .data(data)
        .enter()
        .append('circle')
        .attr('class', 'line-point three-pt-point')
        .attr('cx', d => xScale(d.season))
        .attr('cy', d => yScale(d.three_pt_rate))
        .attr('r', 5)
        .style('cursor', 'pointer')
        .on('mouseover', function(event, d) {
            utils.showTooltip(event, `
                <strong>${d.season} Season</strong><br/>
                3-Point Rate: <span style="color: ${CONFIG.colors.threePt}">${d.three_pt_rate}%</span><br/>
                Total 3PT Attempts: ${utils.formatNumber(d.three_pt_shots)}<br/>
                League Change: ${d.threePtChange ? (d.threePtChange > 0 ? '+' : '') + d.threePtChange.toFixed(1) + '% vs previous year' : 'First year'}
            `);
            d3.select(this).transition().duration(200).attr('r', 8);
        })
        .on('mouseout', function() {
            utils.hideTooltip();
            d3.select(this).transition().duration(200).attr('r', 5);
        });
    
    pointsGroup.selectAll('.mid-range-point')
        .data(data)
        .enter()
        .append('circle')
        .attr('class', 'line-point mid-range-point')
        .attr('cx', d => xScale(d.season))
        .attr('cy', d => yScale(d.mid_range_rate))
        .attr('r', 5)
        .style('cursor', 'pointer')
        .on('mouseover', function(event, d) {
            utils.showTooltip(event, `
                <strong>${d.season} Season</strong><br/>
                Mid-Range Rate: <span style="color: ${CONFIG.colors.midRange}">${d.mid_range_rate}%</span><br/>
                Total Mid-Range: ${utils.formatNumber(d.mid_range_shots)}<br/>
                Mid-Range Decline: ${d.season > 2004 ? 'Down from ' + Math.round(35.65 - d.mid_range_rate + 35.65) + '% in 2004' : 'Baseline year'}
            `);
            d3.select(this).transition().duration(200).attr('r', 8);
        })
        .on('mouseout', function() {
            utils.hideTooltip();
            d3.select(this).transition().duration(200).attr('r', 5);
        });
    
    // Add axis labels
    g.append('text')
        .attr('class', 'axis-label')
        .attr('x', width / 2)
        .attr('y', height + 60)
        .attr('text-anchor', 'middle')
        .style('font-size', '16px')
        .text('Season');
    
    g.append('text')
        .attr('class', 'axis-label')
        .attr('transform', 'rotate(-90)')
        .attr('y', -50)
        .attr('x', -height / 2)
        .attr('text-anchor', 'middle')
        .style('font-size', '16px')
        .text('Percentage of Total Shots');
    
    // Enhanced legend
    const legend = g.append('g')
        .attr('transform', `translate(${width - 200}, 40)`);
    
    const legendData = [
        { label: '3-Point Rate', color: CONFIG.colors.threePt },
        { label: 'Mid-Range Rate', color: CONFIG.colors.midRange }
    ];
    
    const legendItems = legend.selectAll('.legend-item')
        .data(legendData)
        .enter()
        .append('g')
        .attr('class', 'legend-item')
        .attr('transform', (d, i) => `translate(0, ${i * 30})`);
    
    legendItems.append('line')
        .attr('x1', 0)
        .attr('x2', 25)
        .attr('y1', 10)
        .attr('y2', 10)
        .attr('stroke', d => d.color)
        .attr('stroke-width', 4);
    
    legendItems.append('text')
        .attr('x', 35)
        .attr('y', 14)
        .style('font-size', '14px')
        .style('font-weight', '500')
        .style('fill', CONFIG.colors.dark)
        .text(d => d.label);
    
    // Add timeline visualization
    drawTimeline();
    
    // Create secondary panels for this scene
    createSecondaryPanels([
        {
            title: "📊 Steady Growth",
            content: "The three-point rate grew consistently at ~1% per year, with acceleration after 2012 due to analytics adoption."
        },
        {
            title: "🎯 Inflection Point", 
            content: "2014-2017 saw the steepest climb as teams like Golden State proved three-point offense could win championships."
        },
        {
            title: "📈 Current Status",
            content: "By 2024, teams attempt nearly 40% of shots from three - a complete transformation of basketball strategy."
        }
    ]);
}

// Continue with remaining scenes in next part due to size constraints...

function drawScene2_Players() {
    const { svg, g, width, height } = utils.setupSVG('#main-visualization');
    const data = state.data.scene3;
    
    // Find Stephen Curry's data
    const curryData = data.find(p => p.player === 'Stephen Curry')?.seasons || [];
    const leagueData = state.data.scene2;
    
    if (curryData.length === 0) {
        showError('Stephen Curry data not found');
        return;
    }
    
    // Create league average data for comparison
    const leagueAvg = leagueData.map(d => ({
        season: d.season,
        three_pt_rate: d.three_pt_rate
    }));
    
    // Enhanced scales
    const xScale = d3.scaleLinear()
        .domain([2009, 2024])
        .range([0, width]);
    
    const yScale = d3.scaleLinear()
        .domain([0, 70])
        .range([height, 0]);
    
    // Add grid
    const xAxis = g.append('g')
        .attr('class', 'axis')
        .attr('transform', `translate(0, ${height})`)
        .call(utils.getTimeAxis(xScale, width));
    
    const yAxis = g.append('g')
        .attr('class', 'axis')
        .call(d3.axisLeft(yScale).tickFormat(d => d + '%').tickSize(-width));
    
    // Style grid
    yAxis.selectAll('.tick line')
        .attr('stroke', CONFIG.colors.gray)
        .attr('stroke-dasharray', '2,2')
        .attr('opacity', 0.3);
    
    // Line generators
    const curryLine = d3.line()
        .x(d => xScale(d.season))
        .y(d => yScale(d.three_pt_rate))
        .curve(d3.curveMonotoneX);
    
    const leagueLine = d3.line()
        .x(d => xScale(d.season))
        .y(d => yScale(d.three_pt_rate))
        .curve(d3.curveMonotoneX);
    
    // Filter league data for Curry's era
    const leagueFiltered = leagueAvg.filter(d => d.season >= 2009 && d.season <= 2024);
    
    // Draw league average line (dashed)
    g.append('path')
        .datum(leagueFiltered)
        .attr('class', 'mid-range-line')
        .attr('d', leagueLine)
        .style('stroke-dasharray', '8,5')
        .style('stroke-width', 3)
        .style('opacity', 0.7);
    
    // Draw Curry's line (bold)
    g.append('path')
        .datum(curryData)
        .attr('class', 'three-pt-line')
        .attr('d', curryLine)
        .style('stroke-width', 5);
    
    // Add points for Curry with enhanced interactivity
    g.selectAll('.curry-point')
        .data(curryData)
        .enter()
        .append('circle')
        .attr('class', 'line-point three-pt-point')
        .attr('cx', d => xScale(d.season))
        .attr('cy', d => yScale(d.three_pt_rate))
        .attr('r', 6)
        .style('cursor', 'pointer')
        .on('mouseover', function(event, d) {
            utils.showTooltip(event, `
                <strong>Stephen Curry - ${d.season}</strong><br/>
                3-Point Rate: <span style="color: ${CONFIG.colors.threePt}">${d.three_pt_rate}%</span><br/>
                Made 3-Pointers: ${d.made_threes}<br/>
                Total Attempts: ${d.three_pt_shots}<br/>
                Shooting %: ${d.three_pt_percentage}%
            `);
            d3.select(this).transition().duration(200).attr('r', 10);
        })
        .on('mouseout', function() {
            utils.hideTooltip();
            d3.select(this).transition().duration(200).attr('r', 6);
        });
    
    // Add title
    g.append('text')
        .attr('class', 'chart-title')
        .attr('x', width / 2)
        .attr('y', -30)
        .style('font-size', '22px')
        .text('Stephen Curry vs League Average');
    
    // Add axis labels
    g.append('text')
        .attr('class', 'axis-label')
        .attr('x', width / 2)
        .attr('y', height + 50)
        .attr('text-anchor', 'middle')
        .text('Season');
    
    g.append('text')
        .attr('class', 'axis-label')
        .attr('transform', 'rotate(-90)')
        .attr('y', -50)
        .attr('x', -height / 2)
        .attr('text-anchor', 'middle')
        .text('3-Point Rate (% of shots)');
    
    // Enhanced legend
    const legend = g.append('g')
        .attr('transform', `translate(${width - 200}, 40)`);
    
    // Stephen Curry line in legend
    legend.append('line')
        .attr('x1', 0)
        .attr('x2', 30)
        .attr('y1', 10)
        .attr('y2', 10)
        .attr('stroke', CONFIG.colors.threePt)
        .attr('stroke-width', 5);
    
    legend.append('text')
        .attr('x', 40)
        .attr('y', 14)
        .style('font-size', '14px')
        .style('font-weight', '600')
        .text('Stephen Curry');
    
    // League average line in legend
    legend.append('line')
        .attr('x1', 0)
        .attr('x2', 30)
        .attr('y1', 35)
        .attr('y2', 35)
        .attr('stroke', CONFIG.colors.midRange)
        .attr('stroke-width', 3)
        .style('stroke-dasharray', '8,5')
        .style('opacity', 0.7);
    
    legend.append('text')
        .attr('x', 40)
        .attr('y', 39)
        .style('font-size', '14px')
        .style('font-weight', '500')
        .text('League Average');
    
    // Add annotation for record season
    const recordSeason = curryData.find(d => d.season === 2016);
    if (recordSeason) {
        const annotations = [
            {
                note: {
                    title: "Historic 2015-16 Season",
                    label: `${recordSeason.made_threes} made 3-pointers\n${recordSeason.three_pt_rate}% of all shots from three\nShattered previous records`
                },
                x: xScale(recordSeason.season),
                y: yScale(recordSeason.three_pt_rate),
                dy: -100,
                dx: -80
            }
        ];
        
        const makeAnnotations = d3.annotation()
            .annotations(annotations);
        
        g.append('g')
            .attr('class', 'annotation-group')
            .call(makeAnnotations);
    }
    
    createSecondaryPanels([
        {
            title: "🏆 Revolutionary Impact",
            content: "Curry didn't just break records - he fundamentally changed how basketball is played, inspiring teams worldwide to adopt three-point focused strategies."
        },
        {
            title: "📊 Statistical Dominance",
            content: `Curry's peak 3PT rate of ${Math.max(...curryData.map(d => d.three_pt_rate))}% was unprecedented, taking nearly 2 out of every 3 shots from beyond the arc.`
        },
        {
            title: "🎯 Accuracy + Volume",
            content: `Unlike previous shooters, Curry combined extreme volume with elite accuracy, proving that high three-point rates could be both sustainable and effective.`
        }
    ]);
}

function drawScene3_Efficiency() {
    const { svg, g, width, height } = utils.setupSVG('#main-visualization');
    const data = state.data.scene2;
    
    // Enhanced scales
    const xScale = d3.scaleLinear()
        .domain(d3.extent(data, d => d.season))
        .range([0, width]);
    
    const yScale = d3.scaleLinear()
        .domain([46, 58])
        .range([height, 0]);
    
    // Add grid
    const xAxis = g.append('g')
        .attr('class', 'axis')
        .attr('transform', `translate(0, ${height})`)
        .call(utils.getTimeAxis(xScale, width));
    
    const yAxis = g.append('g')
        .attr('class', 'axis')
        .call(d3.axisLeft(yScale).tickFormat(d => d + '%').tickSize(-width));
    
    // Style grid
    yAxis.selectAll('.tick line')
        .attr('stroke', CONFIG.colors.gray)
        .attr('stroke-dasharray', '2,2')
        .attr('opacity', 0.3);
    
    // Line generator
    const efficiencyLine = d3.line()
        .x(d => xScale(d.season))
        .y(d => yScale(d.efg_percentage))
        .curve(d3.curveMonotoneX);
    
    // Add area under curve for visual impact
    const area = d3.area()
        .x(d => xScale(d.season))
        .y0(height)
        .y1(d => yScale(d.efg_percentage))
        .curve(d3.curveMonotoneX);
    
    // Define gradient for area
    const defs = svg.append('defs');
    const gradient = defs.append('linearGradient')
        .attr('id', 'efficiencyGradient')
        .attr('x1', '0%').attr('y1', '0%')
        .attr('x2', '0%').attr('y2', '100%');
    
    gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', CONFIG.colors.efficiency)
        .attr('stop-opacity', 0.4);
        
    gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', CONFIG.colors.efficiency)
        .attr('stop-opacity', 0.1);
    
    // Draw area
    g.append('path')
        .datum(data)
        .attr('class', 'efficiency-area')
        .attr('d', area)
        .attr('fill', 'url(#efficiencyGradient)');
    
    // Draw efficiency line
    g.append('path')
        .datum(data)
        .attr('class', 'efficiency-line')
        .attr('d', efficiencyLine)
        .style('stroke-width', 4);
    
    // Add interactive points
    g.selectAll('.efficiency-point')
        .data(data)
        .enter()
        .append('circle')
        .attr('class', 'line-point efficiency-point')
        .attr('cx', d => xScale(d.season))
        .attr('cy', d => yScale(d.efg_percentage))
        .attr('r', 5)
        .style('cursor', 'pointer')
        .on('mouseover', function(event, d) {
            utils.showTooltip(event, `
                <strong>${d.season} Season</strong><br/>
                Effective FG%: <span style="color: ${CONFIG.colors.efficiency}">${d.efg_percentage}%</span><br/>
                3-Point Rate: ${d.three_pt_rate}%<br/>
                Efficiency Change: ${d.efficiencyChange > 0 ? '+' : ''}${d.efficiencyChange?.toFixed(2) || 0}%
            `);
            d3.select(this).transition().duration(200).attr('r', 8);
        })
        .on('mouseout', function() {
            utils.hideTooltip();
            d3.select(this).transition().duration(200).attr('r', 5);
        });
    
    // Add title
    g.append('text')
        .attr('class', 'chart-title')
        .attr('x', width / 2)
        .attr('y', -30)
        .style('font-size', '22px')
        .text('League Shooting Efficiency Over Time');
    
    // Add axis labels
    g.append('text')
        .attr('class', 'axis-label')
        .attr('x', width / 2)
        .attr('y', height + 50)
        .attr('text-anchor', 'middle')
        .text('Season');
    
    g.append('text')
        .attr('class', 'axis-label')
        .attr('transform', 'rotate(-90)')
        .attr('y', -50)
        .attr('x', -height / 2)
        .attr('text-anchor', 'middle')
        .text('Effective Field Goal Percentage');
    
    // Add annotations for key insights
    const firstYear = data[0];
    const lastYear = data[data.length - 1];
    const improvement = lastYear.efg_percentage - firstYear.efg_percentage;
    
    const annotations = [
        {
            note: {
                title: "Efficiency Gains",
                label: `+${improvement.toFixed(1)} percentage points\nfrom ${firstYear.efg_percentage}% to ${lastYear.efg_percentage}%`
            },
            x: xScale(lastYear.season),
            y: yScale(lastYear.efg_percentage),
            dy: -80,
            dx: -100
        }
    ];
    
    const makeAnnotations = d3.annotation()
        .annotations(annotations);
    
    g.append('g')
        .attr('class', 'annotation-group')
        .call(makeAnnotations);
    
    // Add explanation text
    g.append('text')
        .attr('x', width / 2)
        .attr('y', height + 75)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('fill', CONFIG.colors.gray)
        .text('Effective FG% = (FGM + 0.5 × 3PM) / FGA (accounts for extra value of 3-pointers)');
    
    createSecondaryPanels([
        {
            title: "📈 Steady Improvement",
            content: `League efficiency improved by ${improvement.toFixed(1)} percentage points over 20 years, directly correlating with increased three-point usage.`
        },
        {
            title: "🎯 Mathematical Advantage",
            content: "Three-pointers offer 50% more points than two-pointers, making them mathematically superior even at lower shooting percentages."
        },
        {
            title: "🏆 Strategic Evolution",
            content: "Teams discovered that attempting more three-pointers leads to higher scoring efficiency, better spacing, and more wins."
        }
    ]);
}

function drawScene4_Explorer() {
    console.log('🎨 Initializing Enhanced NBA Explorer...');
    
    // Hide main visualization for enhanced explorer
    d3.select('#main-visualization').style('display', 'none');
    
    // Initialize the advanced explorer
    if (window.nbaAdvancedExplorer) {
        window.nbaAdvancedExplorer.initialize().catch(error => {
            console.error('Failed to initialize enhanced explorer:', error);
            // Fallback to basic explorer
            updateExplorerVisualization();
        });
    } else {
        console.warn('Enhanced explorer not loaded, using basic version');
        updateExplorerVisualization();
    }
}

function updateExplorerVisualization() {
    const { svg, g, width, height } = utils.setupSVG('#explorer-main', 800, 400);
    
    switch (state.viewMode) {
        case 'evolution':
            if (state.analysisType === 'players') {
                drawPlayerEvolution(g, width, height);
            } else {
                drawTeamEvolution(g, width, height);
            }
            break;
        case 'comparison':
            if (state.analysisType === 'players') {
                drawPlayerComparison(g, width, height);
            } else {
                drawTeamComparison(g, width, height);
            }
            break;
        case 'heatmap':
            drawShotHeatmap(g, width, height);
            break;
        case 'performance':
            drawPerformanceDashboard(g, width, height);
            break;
    }
    
    updateEntityStats();
    updatePerformanceMetrics();
}

function drawPlayerEvolution(g, width, height) {
    if (state.selectedPlayers.length === 0) {
        g.append('text')
            .attr('x', width / 2)
            .attr('y', height / 2)
            .attr('text-anchor', 'middle')
            .style('font-size', '18px')
            .style('fill', CONFIG.colors.gray)
            .text('Select a player to view evolution');
        return;
    }

    const player = state.selectedPlayers[0];
    const playerData = getAllPlayerData().find(p => p.player === player);
    if (!playerData) return;
    
    const filteredSeasons = playerData.seasons.filter(s => 
        s.season >= state.startYear && s.season <= state.endYear
    );
    
    // Scales
    const xScale = d3.scaleLinear()
        .domain([state.startYear, state.endYear])
        .range([0, width]);
    
    const yScale = d3.scaleLinear()
        .domain([0, 100])
        .range([height, 0]);
    
    // Create stacked data for area chart
    const stackedData = filteredSeasons.map(d => ({
        season: d.season,
        three_pt_rate: d.three_pt_rate,
        other_rate: 100 - d.three_pt_rate,
        three_pt_shots: d.three_pt_shots,
        made_threes: d.made_threes,
        total_shots: d.total_shots
    }));
    
    // Area generators
    const area3Pt = d3.area()
        .x(d => xScale(d.season))
        .y0(d => yScale(d.other_rate))
        .y1(d => yScale(d.other_rate + d.three_pt_rate))
        .curve(d3.curveMonotoneX);
    
    const areaOther = d3.area()
        .x(d => xScale(d.season))
        .y0(height)
        .y1(d => yScale(d.other_rate))
        .curve(d3.curveMonotoneX);
    
    // Draw areas
    g.append('path')
        .datum(stackedData)
        .attr('class', 'area-other')
        .attr('d', areaOther)
        .style('fill', CONFIG.colors.midRange)
        .style('opacity', 0.7);
    
    g.append('path')
        .datum(stackedData)
        .attr('class', 'area-three-pt')
        .attr('d', area3Pt)
        .style('fill', CONFIG.colors.threePt)
        .style('opacity', 0.7);
    
    // Add axes with proper spacing
    g.append('g')
        .attr('class', 'axis')
        .attr('transform', `translate(0, ${height})`)
        .call(utils.getTimeAxis(xScale, width));
    
    g.append('g')
        .attr('class', 'axis')
        .call(d3.axisLeft(yScale).tickFormat(d => d + '%'));
    
    // Add interactive points
    g.selectAll('.interactive-point')
        .data(stackedData)
        .enter()
        .append('circle')
        .attr('class', 'interactive-point')
        .attr('cx', d => xScale(d.season))
        .attr('cy', d => yScale(d.other_rate + d.three_pt_rate / 2))
        .attr('r', 6)
        .style('fill', 'white')
        .style('stroke', CONFIG.colors.threePt)
        .style('stroke-width', 3)
        .style('cursor', 'pointer')
        .on('mouseover', function(event, d) {
            utils.showTooltip(event, `
                <strong>${player} - ${d.season}</strong><br/>
                3-Point Rate: ${d.three_pt_rate.toFixed(1)}%<br/>
                3-Point Attempts: ${d.three_pt_shots}<br/>
                Made 3-Pointers: ${d.made_threes}<br/>
                Total Shots: ${d.total_shots}
            `);
        })
        .on('mouseout', utils.hideTooltip);
    
    // Add title
    g.append('text')
        .attr('class', 'chart-title')
        .attr('x', width / 2)
        .attr('y', -20)
        .text(`${player}: Shot Profile Evolution`);
}

function drawPlayerComparison(g, width, height) {
    if (state.selectedPlayers.length < 2) {
        g.append('text')
            .attr('x', width / 2)
            .attr('y', height / 2)
            .attr('text-anchor', 'middle')
            .style('font-size', '18px')
            .style('fill', CONFIG.colors.gray)
            .text('Select 2+ players to compare');
        return;
    }

    // Interactive multi-line comparison chart
    const players = state.selectedPlayers.slice(0, 4);
    const playerDataList = players.map(player => 
        getAllPlayerData().find(p => p.player === player)
    ).filter(Boolean);

    if (playerDataList.length === 0) return;

    // Create scales
    const allSeasons = [];
    playerDataList.forEach(player => {
        player.seasons.forEach(season => {
            if (season.season >= state.startYear && season.season <= state.endYear) {
                allSeasons.push(season);
            }
        });
    });

    const xScale = d3.scaleLinear()
        .domain([state.startYear, state.endYear])
        .range([0, width]);
    
    const yScale = d3.scaleLinear()
        .domain([0, Math.max(60, d3.max(allSeasons, d => d.three_pt_rate))])
        .range([height, 0]);

    // Add grid and axes
    g.append('g')
        .attr('class', 'axis')
        .attr('transform', `translate(0, ${height})`)
        .call(utils.getTimeAxis(xScale, width));
    
    g.append('g')
        .attr('class', 'axis')
        .call(d3.axisLeft(yScale).tickFormat(d => d + '%').tickSize(-width))
        .selectAll('.tick line')
        .attr('stroke', CONFIG.colors.gray)
        .attr('stroke-dasharray', '2,2')
        .attr('opacity', 0.3);

    const colors = [CONFIG.colors.threePt, CONFIG.colors.midRange, CONFIG.colors.efficiency, CONFIG.colors.accent];
    
    // Draw lines for each player
    playerDataList.forEach((playerData, index) => {
        const filteredSeasons = playerData.seasons.filter(s => 
            s.season >= state.startYear && s.season <= state.endYear
        );

        const line = d3.line()
            .x(d => xScale(d.season))
            .y(d => yScale(d.three_pt_rate))
            .curve(d3.curveMonotoneX);

        // Draw line
        g.append('path')
            .datum(filteredSeasons)
            .attr('d', line)
            .attr('fill', 'none')
            .attr('stroke', colors[index])
            .attr('stroke-width', 3)
            .style('opacity', 0.8);

        // Add interactive points
        g.selectAll(`.player-points-${index}`)
            .data(filteredSeasons)
            .enter()
            .append('circle')
            .attr('class', `player-points-${index}`)
            .attr('cx', d => xScale(d.season))
            .attr('cy', d => yScale(d.three_pt_rate))
            .attr('r', 5)
            .attr('fill', colors[index])
            .attr('stroke', 'white')
            .attr('stroke-width', 2)
            .style('cursor', 'pointer')
            .on('mouseover', function(event, d) {
                utils.showTooltip(event, `
                    <strong>${playerData.player} - ${d.season}</strong><br/>
                    3-Point Rate: <span style="color: ${colors[index]}">${d.three_pt_rate}%</span><br/>
                    Made 3-Pointers: ${d.made_threes}<br/>
                    Shooting %: ${d.three_pt_percentage}%<br/>
                    Total Shots: ${d.total_shots}
                `);
                d3.select(this).transition().duration(200).attr('r', 8);
            })
            .on('mouseout', function() {
                utils.hideTooltip();
                d3.select(this).transition().duration(200).attr('r', 5);
            });
    });

    // Add title
    g.append('text')
        .attr('class', 'chart-title')
        .attr('x', width / 2)
        .attr('y', -20)
        .text('Player Three-Point Rate Comparison');

    // Add legend
    const legend = g.append('g')
        .attr('transform', `translate(${width - 150}, 20)`);
        
    playerDataList.forEach((playerData, i) => {
        const legendItem = legend.append('g')
            .attr('transform', `translate(0, ${i * 25})`);
            
        legendItem.append('line')
            .attr('x1', 0)
            .attr('x2', 20)
            .attr('y1', 10)
            .attr('y2', 10)
            .attr('stroke', colors[i])
            .attr('stroke-width', 3);
            
        legendItem.append('text')
            .attr('x', 25)
            .attr('y', 14)
            .style('font-size', '12px')
            .style('font-weight', '500')
            .text(playerData.player);
    });


}

function drawShotHeatmap(g, width, height) {
    // Implementation for shot distribution heatmap
    g.append('text')
        .attr('x', width / 2)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .style('font-size', '18px')
        .text('Shot Distribution Heatmap - Coming Soon');
}

// Enhanced search controls
function initializeSearchControls() {
    const playerSearch = document.getElementById('player-search');
    const teamSearch = document.getElementById('team-search');
    const playerDropdown = document.getElementById('player-dropdown');
    const teamDropdown = document.getElementById('team-dropdown');

    // Safe event listener binding with existence checks
    if (playerSearch && playerDropdown) {
        playerSearch.addEventListener('input', (e) => {
            showSearchResults(e.target.value, state.availablePlayers, playerDropdown, 'player');
        });
        console.log('✅ Player search controls initialized');
    } else {
        console.warn('⚠️ Player search elements not found - explorer may not be active');
    }

    if (teamSearch && teamDropdown) {
        teamSearch.addEventListener('input', (e) => {
            showSearchResults(e.target.value, state.availableTeams, teamDropdown, 'team');
        });
        console.log('✅ Team search controls initialized');
    } else {
        console.warn('⚠️ Team search elements not found - explorer may not be active');
    }

    // Document click handler (safe since it's on document) - only if dropdowns exist
    if (playerDropdown && teamDropdown) {
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.search-container')) {
                playerDropdown.classList.remove('show');
                teamDropdown.classList.remove('show');
            }
        });
        console.log('✅ Search dropdown handlers initialized');
    }
}

function showSearchResults(query, items, dropdown, type) {
    if (query.length < 1) {
        dropdown.classList.remove('show');
        return;
    }

    const filtered = items.filter(item => 
        item.toLowerCase().includes(query.toLowerCase())
    );

    dropdown.innerHTML = '';
    if (filtered.length > 0) {
        filtered.slice(0, 10).forEach(item => {
            const div = document.createElement('div');
            div.className = 'dropdown-item';
            div.textContent = item;
            div.addEventListener('click', () => selectItem(item, type));
            dropdown.appendChild(div);
        });
        dropdown.classList.add('show');
    } else {
        dropdown.classList.remove('show');
    }
}

function selectItem(item, type) {
    if (type === 'player') {
        if (!state.selectedPlayers.includes(item)) {
            state.selectedPlayers.push(item);
            updateSelectedItems('players');
        }
        // Safe element access
        const playerSearch = document.getElementById('player-search');
        const playerDropdown = document.getElementById('player-dropdown');
        if (playerSearch) playerSearch.value = '';
        if (playerDropdown) playerDropdown.classList.remove('show');
    } else {
        if (!state.selectedTeams.includes(item)) {
            state.selectedTeams.push(item);
            updateSelectedItems('teams');
        }
        // Safe element access
        const teamSearch = document.getElementById('team-search');
        const teamDropdown = document.getElementById('team-dropdown');
        if (teamSearch) teamSearch.value = '';
        if (teamDropdown) teamDropdown.classList.remove('show');
    }
    
    if (state.currentScene === 4) updateExplorerVisualization();
}

function updateSelectedItems(type) {
    const container = type === 'players' ? 
        document.getElementById('selected-players') : 
        document.getElementById('selected-teams');
    
    // Check if container exists
    if (!container) {
        console.warn(`Container for ${type} not found`);
        return;
    }
    
    const items = type === 'players' ? state.selectedPlayers : state.selectedTeams;
    
    container.innerHTML = '';
    items.forEach(item => {
        const div = document.createElement('div');
        div.className = 'selected-item';
        div.dataset[type.slice(0, -1)] = item;
        div.innerHTML = `
            <span>${item}</span>
            <button class="remove-btn" onclick="removeItem('${item}', '${type}')">×</button>
        `;
        container.appendChild(div);
    });
}

function removeItem(item, type) {
    if (type === 'players') {
        state.selectedPlayers = state.selectedPlayers.filter(p => p !== item);
        updateSelectedItems('players');
    } else {
        state.selectedTeams = state.selectedTeams.filter(t => t !== item);
        updateSelectedItems('teams');
    }
    
    if (state.currentScene === 4) updateExplorerVisualization();
}

function toggleAnalysisControls() {
    const playerControls = document.getElementById('player-controls');
    const teamControls = document.getElementById('team-controls');
    
    if (state.analysisType === 'players') {
        playerControls.style.display = 'block';
        teamControls.style.display = 'none';
    } else {
        playerControls.style.display = 'none';
        teamControls.style.display = 'block';
    }
}

function updateChartTitle() {
    const title = document.getElementById('chart-title');
    const analysisType = state.analysisType === 'players' ? 'Player' : 'Team';
    const viewMode = state.viewMode.charAt(0).toUpperCase() + state.viewMode.slice(1);
    title.textContent = `${analysisType} ${viewMode} Analysis`;
}

function updateEntityStats() {
    const statsTitle = document.getElementById('stats-title');
    const statsContainer = document.getElementById('entity-stats');
    
    if (state.analysisType === 'players') {
        statsTitle.textContent = 'Player Statistics';
        updatePlayerStats(statsContainer);
    } else {
        statsTitle.textContent = 'Team Statistics';
        updateTeamStats(statsContainer);
    }
}

function updatePlayerStats(container) {
    if (state.selectedPlayers.length === 0) {
        container.innerHTML = '<p>No players selected</p>';
        return;
    }

    const player = state.selectedPlayers[0]; // Show stats for first selected player
    const playerData = getAllPlayerData().find(p => p.player === player);
    
    if (!playerData) {
        container.innerHTML = '<p>Player data not found</p>';
        return;
    }
    
    const stats = playerData.seasons;
    const careerStats = {
        totalSeasons: stats.length,
        totalThrees: stats.reduce((sum, s) => sum + s.made_threes, 0),
        totalAttempts: stats.reduce((sum, s) => sum + s.three_pt_shots, 0),
        avgRate: (stats.reduce((sum, s) => sum + s.three_pt_rate, 0) / stats.length).toFixed(1),
        peakRate: Math.max(...stats.map(s => s.three_pt_rate)).toFixed(1)
    };
    
    const accuracy = ((careerStats.totalThrees / careerStats.totalAttempts) * 100).toFixed(1);
    
    container.innerHTML = `
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-size: 14px;">
            <div><strong>Player:</strong> ${player}</div>
            <div><strong>Seasons:</strong> ${careerStats.totalSeasons}</div>
            <div><strong>Made 3s:</strong> ${careerStats.totalThrees}</div>
            <div><strong>Attempts:</strong> ${careerStats.totalAttempts}</div>
            <div><strong>Accuracy:</strong> ${accuracy}%</div>
            <div><strong>Avg Rate:</strong> ${careerStats.avgRate}%</div>
            <div><strong>Peak Rate:</strong> ${careerStats.peakRate}%</div>
            <div><strong>Years:</strong> ${Math.min(...stats.map(s => s.season))}-${Math.max(...stats.map(s => s.season))}</div>
        </div>
    `;
}

function updateTeamStats(container) {
    if (state.selectedTeams.length === 0) {
        container.innerHTML = '<p>No teams selected</p>';
        return;
    }

    const team = state.selectedTeams[0]; // Show stats for first selected team
    const teamData = state.data.scene4.team_data?.find(t => t.team === team);
    
    if (!teamData) {
        container.innerHTML = '<p>Team data not found</p>';
        return;
    }
    
    const stats = teamData.seasons;
    const teamStats = {
        totalSeasons: stats.length,
        avgRate: (stats.reduce((sum, s) => sum + s.three_pt_rate, 0) / stats.length).toFixed(1),
        peakRate: Math.max(...stats.map(s => s.three_pt_rate)).toFixed(1),
        totalWins: stats.reduce((sum, s) => sum + s.wins, 0),
        playoffSeasons: stats.filter(s => s.playoffs).length,
        avgWins: (stats.reduce((sum, s) => sum + s.wins, 0) / stats.length).toFixed(1)
    };
    
    container.innerHTML = `
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-size: 14px;">
            <div><strong>Team:</strong> ${team}</div>
            <div><strong>Seasons:</strong> ${teamStats.totalSeasons}</div>
            <div><strong>Total Wins:</strong> ${teamStats.totalWins}</div>
            <div><strong>Avg Wins/Season:</strong> ${teamStats.avgWins}</div>
            <div><strong>Playoff Seasons:</strong> ${teamStats.playoffSeasons}</div>
            <div><strong>Avg 3PT Rate:</strong> ${teamStats.avgRate}%</div>
            <div><strong>Peak 3PT Rate:</strong> ${teamStats.peakRate}%</div>
            <div><strong>Years:</strong> ${Math.min(...stats.map(s => s.season))}-${Math.max(...stats.map(s => s.season))}</div>
        </div>
    `;
}

function updatePerformanceMetrics() {
    const container = document.getElementById('performance-metrics');
    
    if (state.analysisType === 'players' && state.selectedPlayers.length > 0) {
        updatePlayerPerformanceMetrics(container);
    } else if (state.analysisType === 'teams' && state.selectedTeams.length > 0) {
        updateTeamPerformanceMetrics(container);
    } else {
        container.innerHTML = '<p>Select entities to view metrics</p>';
    }
}

function updatePlayerPerformanceMetrics(container) {
    const player = state.selectedPlayers[0];
    const playerData = getAllPlayerData().find(p => p.player === player);
    
    if (!playerData) return;
    
    const filteredSeasons = playerData.seasons.filter(s => 
        s.season >= state.startYear && s.season <= state.endYear
    );
    
    if (filteredSeasons.length === 0) {
        container.innerHTML = '<p>No data for selected years</p>';
        return;
    }
    
    const totalAttempts = filteredSeasons.reduce((sum, s) => sum + s.three_pt_shots, 0);
    const totalMade = filteredSeasons.reduce((sum, s) => sum + s.made_threes, 0);
    const avgRate = (filteredSeasons.reduce((sum, s) => sum + s.three_pt_rate, 0) / filteredSeasons.length).toFixed(1);
    const accuracy = ((totalMade / totalAttempts) * 100).toFixed(1);
    
    container.innerHTML = `
        <div class="metric-item">
            <span class="metric-label">Total 3PT</span>
            <span class="metric-value">${totalMade}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Accuracy</span>
            <span class="metric-value">${accuracy}%</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Avg Rate</span>
            <span class="metric-value">${avgRate}%</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Seasons</span>
            <span class="metric-value">${filteredSeasons.length}</span>
        </div>
    `;
}

function updateTeamPerformanceMetrics(container) {
    const team = state.selectedTeams[0];
    const teamData = state.data.scene4.team_data?.find(t => t.team === team);
    
    if (!teamData) return;
    
    let filteredSeasons = teamData.seasons.filter(s => 
        s.season >= state.startYear && s.season <= state.endYear
    );
    
    // Apply filters
    if (state.filters.playoffOnly) {
        filteredSeasons = filteredSeasons.filter(s => s.playoffs);
    }
    if (state.filters.winningTeams) {
        filteredSeasons = filteredSeasons.filter(s => s.wins >= 41);
    }
    
    if (filteredSeasons.length === 0) {
        container.innerHTML = '<p>No data for selected years/filters</p>';
        return;
    }
    
    const totalShots = filteredSeasons.reduce((sum, s) => sum + s.total_shots, 0);
    const totalThreeAttempts = filteredSeasons.reduce((sum, s) => sum + s.three_pt_shots, 0);
    const totalThreeMade = filteredSeasons.reduce((sum, s) => sum + s.three_pt_made, 0);
    const avgRate = (filteredSeasons.reduce((sum, s) => sum + s.three_pt_rate, 0) / filteredSeasons.length).toFixed(1);
    const threePointAccuracy = totalThreeAttempts > 0 ? ((totalThreeMade / totalThreeAttempts) * 100).toFixed(1) : '0.0';
    
    container.innerHTML = `
        <div class="metric-item">
            <span class="metric-label">Total Shots</span>
            <span class="metric-value">${totalShots.toLocaleString()}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">3PT Made</span>
            <span class="metric-value">${totalThreeMade.toLocaleString()}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">3PT Accuracy</span>
            <span class="metric-value">${threePointAccuracy}%</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Avg 3PT Rate</span>
            <span class="metric-value">${avgRate}%</span>
        </div>
    `;
}

function getAllPlayerData() {
    const scene3Data = state.data.scene3 || [];
    const comprehensiveData = state.data.scene4.player_data || [];
    return [...scene3Data, ...comprehensiveData];
}

// Timeline visualization for Scene 1
function drawTimeline() {
    const { svg, g, width, height } = utils.setupSVG('#timeline-viz', 900, 100);
    const data = state.data.scene2;
    
    const xScale = d3.scaleLinear()
        .domain(d3.extent(data, d => d.season))
        .range([0, width]);
    
    const radiusScale = d3.scaleLinear()
        .domain(d3.extent(data, d => d.three_pt_rate))
        .range([3, 15]);
    
    // Timeline line
    g.append('line')
        .attr('x1', 0)
        .attr('x2', width)
        .attr('y1', height/2)
        .attr('y2', height/2)
        .attr('stroke', CONFIG.colors.gray)
        .attr('stroke-width', 2);
    
    // Timeline points
    g.selectAll('.timeline-point')
        .data(data)
        .enter()
        .append('circle')
        .attr('class', 'timeline-point')
        .attr('cx', d => xScale(d.season))
        .attr('cy', height/2)
        .attr('r', d => radiusScale(d.three_pt_rate))
        .attr('fill', CONFIG.colors.threePt)
        .attr('stroke', CONFIG.colors.background)
        .attr('stroke-width', 2)
        .style('cursor', 'pointer')
        .on('mouseover', function(event, d) {
            utils.showTooltip(event, `
                <strong>${d.season}</strong><br/>
                3PT Rate: ${d.three_pt_rate}%<br/>
                Total Shots: ${utils.formatNumber(d.total_shots)}
            `);
        })
        .on('mouseout', utils.hideTooltip);
}

// Utility functions
function createSecondaryPanels(panels) {
    const container = document.getElementById('secondary-panels');
    container.innerHTML = '';
    
    panels.forEach((panel, index) => {
        const panelDiv = document.createElement('div');
        panelDiv.className = 'secondary-panel fade-in';
        panelDiv.style.animationDelay = `${index * 100}ms`;
        
        panelDiv.innerHTML = `
            <h3 class="panel-title">${panel.title}</h3>
            <div class="panel-content">${panel.content}</div>
        `;
        
        container.appendChild(panelDiv);
    });
}

function showLoading() {
    document.getElementById('loading-overlay').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading-overlay').classList.add('hidden');
}

function showError(message) {
    console.error(message);
    hideLoading();
    
    // Create error display
    const errorDiv = document.createElement('div');
    errorDiv.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: #fee2e2;
        border: 1px solid #fca5a5;
        color: #991b1b;
        padding: 20px;
        border-radius: 8px;
        max-width: 500px;
        text-align: center;
        z-index: 10000;
        font-family: Inter, sans-serif;
    `;
    errorDiv.innerHTML = `
        <h3 style="margin: 0 0 10px 0; color: #991b1b;">⚠️ Error Loading Visualization</h3>
        <p style="margin: 0 0 15px 0;">${message}</p>
        <button onclick="location.reload()" style="
            background: #dc2626; 
            color: white; 
            border: none; 
            padding: 8px 16px; 
            border-radius: 4px; 
            cursor: pointer;
        ">🔄 Refresh Page</button>
    `;
    document.body.appendChild(errorDiv);
}

function toggleTimelineAnimation() {
    const btn = document.getElementById('play-btn');
    
    if (state.isPlaying) {
        clearInterval(state.animationFrame);
        btn.textContent = '▶️ Play Timeline';
        state.isPlaying = false;
    } else {
        btn.textContent = '⏸️ Pause Timeline';
        state.isPlaying = true;
        
        let currentYear = 2004;
        state.animationFrame = setInterval(() => {
            document.getElementById('timeline-year').textContent = currentYear;
            
            // Update visualization for current year
            highlightTimelineYear(currentYear);
            
            currentYear++;
            if (currentYear > 2024) {
                currentYear = 2004;
            }
        }, 500);
    }
}

function highlightTimelineYear(year) {
    d3.selectAll('.timeline-point')
        .attr('stroke-width', d => d.season === year ? 4 : 2)
        .attr('stroke', d => d.season === year ? CONFIG.colors.accent : CONFIG.colors.background);
}

// Export for debugging
window.NBADashboard = {
    state,
    CONFIG,
    updateScene,
    changeScene,
    utils
};