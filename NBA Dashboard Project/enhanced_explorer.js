
class NBAAdvancedExplorer {
    constructor() {
        this.data = null;
        this.topShooters = null;
        this.selectedTeams = new Set();
        this.selectedPlayers = new Set();
        this.currentView = 'teams';
        this.timeRange = { start: 2004, end: 2024 };
        this.colors = {
            primary: '#ea580c',
            secondary: '#1e40af',
            success: '#16a34a',
            accent: '#f59e0b',
            gray: '#64748b'
        };
    }

    async initialize() {
        try {
            console.log('🚀 Starting enhanced explorer initialization...');
            
            await this.loadMasterData();
            console.log('✅ Data loaded, setting up UI...');
            
            this.setupAdvancedUI();
            console.log('✅ UI setup complete, updating visualization...');
            
            this.updateVisualization();
            console.log('🎉 Enhanced explorer fully initialized!');

        } catch (error) {
            console.error('Failed to initialize enhanced explorer:', error);
            console.error('❌ Initialization error details:', error.message);
            console.error('❌ Initialization stack:', error.stack);
            this.showError(`Failed to load master dataset: ${error.message}`);
        }
    }

    async loadMasterData() {
        try {
            const [enhancedData, topShooters, teamConferences] = await Promise.all([
                d3.json('data/scene4_data_enhanced.json'),
                d3.json('data/top_30_three_point_shooters.json'),
                d3.json('data/teams_by_conference.json')
            ]);
            
            this.data = enhancedData;
            this.data.team_conferences = teamConferences;
            this.topShooters = topShooters;
            
            console.log('✅ Master data loaded successfully:', {
                player_data: this.data.player_data?.length || 0,
                team_data: this.data.team_data?.length || 0,
                team_conferences: Object.keys(this.data.team_conferences || {}).length,
                topShooters: this.topShooters?.length || 0
            });
            

        } catch (error) {
            console.error('❌ Error loading master data:', error);
            console.error('❌ Detailed error:', error.message);
            console.error('❌ Stack trace:', error.stack);
            throw new Error(`Could not load master dataset: ${error.message}`);
        }
    }

    setupAdvancedUI() {
        const explorerContainer = document.getElementById('explorer-dashboard');
        explorerContainer.innerHTML = this.createFullUIHTML();
        this.bindEventListeners();
        this.populateTeamFilters();
        this.populateTopShooters();
        this.populatePlayerList();
        
        // Set initial view based on main UI state
        if (window.state && window.state.analysisType) {
            this.currentView = window.state.analysisType === 'players' ? 'players' : 'teams';
            console.log(`🎯 Set initial view from main UI state: ${this.currentView} (analysisType: ${window.state.analysisType})`);
        } else {
            console.log(`🎯 Using default view: ${this.currentView} (no main UI state found)`);
        }
        
        // Bind team selection listeners after UI is populated
        this.bindTeamSelectionListeners();
        // Bind player selection listeners after UI is populated
        this.bindShooterSelectionListeners();
        this.updateViewSections();
    }

    createFullUIHTML() {
        return `
            <div style="display: flex; gap: 20px; width: 100%; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 12px; padding: 20px; box-sizing: border-box;">
                <!-- Control Panel -->
                <div style="width: 350px; background: white; border-radius: 12px; padding: 20px; border: 1px solid #e2e8f0; box-sizing: border-box; height: fit-content; max-height: 800px; overflow-y: auto;">
                    ${this.createControlPanelHTML()}
                </div>
                
                <!-- Main Visualization -->
                <div style="flex: 1; background: white; border-radius: 12px; padding: 20px; border: 1px solid #e2e8f0; box-sizing: border-box; display: flex; flex-direction: column;">
                    ${this.createVisualizationAreaHTML()}
                </div>
                
                <!-- Statistics Panel -->
                <div style="width: 300px; background: white; border-radius: 12px; padding: 20px; border: 1px solid #e2e8f0; box-sizing: border-box; height: fit-content; max-height: 800px; overflow-y: auto;">
                    ${this.createStatsPanelHTML()}
                </div>
            </div>
        `;
    }

    createControlPanelHTML() {
        return `
            <div style="margin-bottom: 24px;">
                <h3 style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 0 0 12px 0; padding-bottom: 8px; border-bottom: 2px solid #ea580c;">🎯 Analysis Type</h3>
                <div style="display: grid; grid-template-columns: 1fr; gap: 8px;">
                    <button class="analysis-type-btn active" data-type="teams" style="padding: 12px 16px; background: linear-gradient(135deg, #ea580c, #dc2626); color: white; border: 2px solid #ea580c; border-radius: 8px; cursor: pointer; transition: all 0.3s ease; font-size: 13px; font-weight: 500; text-align: left;">
                        🏟️ Team Trends
                    </button>
                    <button class="analysis-type-btn" data-type="players" style="padding: 12px 16px; background: #f1f5f9; border: 2px solid #e2e8f0; border-radius: 8px; cursor: pointer; transition: all 0.3s ease; font-size: 13px; font-weight: 500; text-align: left; color: #475569;">
                        🏀 Player Comparison
                    </button>
                    <button class="analysis-type-btn" data-type="shooters" style="padding: 12px 16px; background: #f1f5f9; border: 2px solid #e2e8f0; border-radius: 8px; cursor: pointer; transition: all 0.3s ease; font-size: 13px; font-weight: 500; text-align: left; color: #475569;">
                        🎯 Elite Shooters
                    </button>
                </div>
            </div>

            <div id="team-filter-section" style="margin-bottom: 24px;">
                <h3 style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 0 0 12px 0; padding-bottom: 8px; border-bottom: 2px solid #ea580c;">🏟️ Team Selection</h3>
                <div style="display: flex; gap: 4px; margin-bottom: 12px;">
                    <button class="conference-tab active" data-conference="Eastern" style="flex: 1; padding: 8px 12px; background: #ea580c; color: white; border: 2px solid #ea580c; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 500;">
                        🌅 Eastern
                    </button>
                    <button class="conference-tab" data-conference="Western" style="flex: 1; padding: 8px 12px; background: #f1f5f9; color: #64748b; border: 2px solid #e2e8f0; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 500;">
                        🌄 Western
                    </button>
                </div>
                <div id="team-divisions-container" style="max-height: 350px; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px;"></div>
            </div>

            <div id="player-filter-section" style="display: none; margin-bottom: 24px;">
                <h3 style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 0 0 12px 0; padding-bottom: 8px; border-bottom: 2px solid #ea580c;">🏀 Top 30 Player Careers (Highest 3pt %)</h3>
                <div id="player-list-container" style="max-height: 350px; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px;"></div>
                <div style="margin-top: 16px;">
                    <h4 style="font-size: 14px; font-weight: 600; color: #374151; margin: 0 0 8px 0;">Selected Players</h4>
                    <div id="selected-players-list" style="display: flex; flex-direction: column; gap: 6px;"></div>
                </div>
            </div>

            <div id="shooters-section" style="display: none; margin-bottom: 24px;">
                <h3 style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 0 0 12px 0; padding-bottom: 8px; border-bottom: 2px solid #ea580c;">🎯 Top 30 Elite Shooters</h3>
                <div id="top-shooters-list" style="max-height: 350px; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px;"></div>
            </div>

            <div style="margin-bottom: 24px;">
                <h3 style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 0 0 12px 0; padding-bottom: 8px; border-bottom: 2px solid #ea580c;">📅 Time Range</h3>
                <div style="display: grid; grid-template-columns: 1fr; gap: 6px; margin-bottom: 16px;">
                    <button class="era-btn" data-era="2004-2008" style="padding: 10px 14px; background: #f8fafc; border: 2px solid #e2e8f0; border-radius: 8px; cursor: pointer; font-size: 12px; font-weight: 500; color: #475569;">Early Era (2004-2008)</button>
                    <button class="era-btn" data-era="2009-2014" style="padding: 10px 14px; background: #f8fafc; border: 2px solid #e2e8f0; border-radius: 8px; cursor: pointer; font-size: 12px; font-weight: 500; color: #475569;">Analytics Rise (2009-2014)</button>
                    <button class="era-btn" data-era="2015-2020" style="padding: 10px 14px; background: #f8fafc; border: 2px solid #e2e8f0; border-radius: 8px; cursor: pointer; font-size: 12px; font-weight: 500; color: #475569;">Curry Era (2015-2020)</button>
                    <button class="era-btn active" data-era="2021-2024" style="padding: 10px 14px; background: #ea580c; color: white; border: 2px solid #ea580c; border-radius: 8px; cursor: pointer; font-size: 12px; font-weight: 500;">Modern NBA (2021-2024)</button>
                    <button class="era-btn" data-era="2004-2024" style="padding: 10px 14px; background: #f8fafc; border: 2px solid #e2e8f0; border-radius: 8px; cursor: pointer; font-size: 12px; font-weight: 500; color: #475569;">Full History</button>
                </div>
                <div style="display: grid; gap: 12px;">
                    <div>
                        <label style="font-weight: 600; color: #ea580c; font-size: 14px;">Start: <span id="start-year-display">2021</span></label>
                        <input type="range" id="start-year-slider" min="2004" max="2024" value="2021" style="width: 100%; height: 6px; background: #e2e8f0; border-radius: 3px; outline: none; -webkit-appearance: none;">
                    </div>
                    <div>
                        <label style="font-weight: 600; color: #ea580c; font-size: 14px;">End: <span id="end-year-display">2024</span></label>
                        <input type="range" id="end-year-slider" min="2004" max="2024" value="2024" style="width: 100%; height: 6px; background: #e2e8f0; border-radius: 3px; outline: none; -webkit-appearance: none;">
                    </div>
                </div>
            </div>
        `;
    }

    createVisualizationAreaHTML() {
        return `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 2px solid #f1f5f9;">
                <h2 id="advanced-chart-title" style="font-size: 20px; font-weight: 700; color: #0f172a; margin: 0;">Team Three-Point Trends</h2>
                <div style="display: flex; gap: 12px;">
                    <div id="chart-option-buttons" style="display: flex; gap: 6px;">
                        <!-- Buttons will be dynamically populated based on view -->
                    </div>
                </div>
            </div>
            <div style="flex: 1; position: relative; min-height: 500px; height: 500px; margin: 10px 0;">
                <svg id="advanced-main-chart" style="width: 100%; height: 100%; background: #fafbfc; border-radius: 8px; display: block;"></svg>
            </div>
            <div id="chart-insights" style="margin-top: 16px; padding: 16px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;"></div>
        `;
    }

    createStatsPanelHTML() {
        return `
            <h3 style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 0 0 16px 0; padding-bottom: 8px; border-bottom: 2px solid #ea580c;">📊 Live Statistics</h3>
            <div style="margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px; background: #f8fafc; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #ea580c;">
                    <div>
                        <div id="selected-count" style="font-size: 18px; font-weight: 700; color: #ea580c;">0</div>
                        <div style="font-size: 12px; color: #64748b; font-weight: 500;">Teams Selected</div>
                    </div>
                </div>
            </div>
            
            <div style="margin-bottom: 20px;">
                <h4 style="font-size: 14px; font-weight: 600; color: #374151; margin: 0 0 8px 0;">📋 Quick Stats</h4>
                <div id="quick-stats-content"></div>
            </div>
            
            <div style="padding: 16px; background: linear-gradient(135deg, #f1f5f9, #e2e8f0); border-radius: 8px; border: 1px solid #cbd5e1;">
                <h4 style="font-size: 14px; font-weight: 600; color: #374151; margin: 0 0 8px 0;">🏀 League Context</h4>
                <div id="league-context-content">
                    <p style="font-size: 13px; color: #374151; line-height: 1.4; margin: 0;">Select teams or players to see contextual statistics and league rankings.</p>
                </div>
            </div>
        `;
    }
}

// Initialize when needed
window.nbaAdvancedExplorer = new NBAAdvancedExplorer();// Enhanced NBA Explorer - Additional Functions

// Add these methods to the NBAAdvancedExplorer class
Object.assign(NBAAdvancedExplorer.prototype, {
    
    populateTeamFilters() {
        const container = document.getElementById('team-divisions-container');
        const conferences = this.data.team_conferences;
        this.renderConferenceDivisions(container, conferences['Eastern Conference'], 'Eastern');
        this.updateSelectedCount();
    },

    renderConferenceDivisions(container, divisions, conference) {
        container.innerHTML = `
            <div id="${conference.toLowerCase()}-divisions" style="display: grid; grid-template-columns: 1fr; gap: 12px;">
                ${Object.entries(divisions).map(([divisionName, teams]) => `
                    <div>
                        <h4 style="font-size: 12px; font-weight: 600; color: #0f172a; margin: 0 0 8px 0; padding: 6px; background: white; border-radius: 4px; text-align: center;">${divisionName}</h4>
                        <div style="display: flex; flex-direction: column; gap: 4px;">
                            ${teams.map(team => `
                                <div class="team-option ${this.selectedTeams.has(team.team) ? 'selected' : ''}" 
                                     data-team="${team.team}"
                                     style="padding: 6px 10px; background: white; border: 2px solid ${this.selectedTeams.has(team.team) ? '#ea580c' : 'transparent'}; border-radius: 4px; font-size: 11px; cursor: pointer; transition: all 0.2s ease; text-align: center; ${this.selectedTeams.has(team.team) ? 'background: #ea580c; color: white;' : ''}">
                                    ${team.team}
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    },

    populateTopShooters() {
        const container = document.getElementById('top-shooters-list');
        container.innerHTML = this.topShooters.map((shooter, index) => `
            <div class="player-card ${this.selectedPlayers.has(shooter.player) ? 'selected' : ''}" 
                 data-player="${shooter.player}"
                 style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: ${this.selectedPlayers.has(shooter.player) ? 'linear-gradient(135deg, #ea580c, #dc2626)' : '#f8fafc'}; color: ${this.selectedPlayers.has(shooter.player) ? 'white' : 'black'}; border-radius: 6px; font-size: 12px; font-weight: 500; margin-bottom: 6px; cursor: pointer;">
                <div>
                    <strong>#${index + 1} ${shooter.player}</strong>
                    <div style="font-size: 10px; opacity: 0.8;">
                        ${shooter.career_three_pt_percentage}% • ${shooter.years_active}
                    </div>
                </div>
                <div style="font-size: 10px;">
                    ${shooter.career_three_pt_made}/${shooter.career_three_pt_attempts}
                </div>
            </div>
        `).join('');
    },

    populatePlayerList() {
        const container = document.getElementById('player-list-container');
        
        if (!container || !this.topShooters) return;
        
        container.innerHTML = this.topShooters.map((shooter, index) => `
            <div class="player-list-option ${this.selectedPlayers.has(shooter.player) ? 'selected' : ''}" 
                 data-player="${shooter.player}"
                 style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: ${this.selectedPlayers.has(shooter.player) ? '#ea580c' : 'white'}; color: ${this.selectedPlayers.has(shooter.player) ? 'white' : 'black'}; border: 2px solid ${this.selectedPlayers.has(shooter.player) ? '#ea580c' : '#e2e8f0'}; border-radius: 6px; font-size: 12px; font-weight: 500; margin-bottom: 4px; cursor: pointer; transition: all 0.2s ease;">
                <div>
                    <strong>#${index + 1} ${shooter.player}</strong>
                    <div style="font-size: 10px; opacity: 0.8;">
                        ${shooter.career_three_pt_percentage}% career • ${shooter.years_active}
                    </div>
                </div>
                <div style="font-size: 10px; font-weight: 600;">
                    ${shooter.career_three_pt_made}/${shooter.career_three_pt_attempts}
                </div>
            </div>
        `).join('');
        
        console.log('✅ populatePlayerList completed, container innerHTML length:', container.innerHTML.length);
    },

    bindEventListeners() {
        // Analysis type buttons
        document.querySelectorAll('.analysis-type-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.analysis-type-btn').forEach(b => {
                    b.classList.remove('active');
                    b.style.background = '#f1f5f9';
                    b.style.color = '#475569';
                    b.style.border = '2px solid #e2e8f0';
                });
                e.target.classList.add('active');
                e.target.style.background = 'linear-gradient(135deg, #ea580c, #dc2626)';
                e.target.style.color = 'white';
                e.target.style.border = '2px solid #ea580c';
                
                this.currentView = e.target.dataset.type;
                this.updateViewSections();
                this.updateVisualization();
            });
        });

        // Conference tabs
        document.querySelectorAll('.conference-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                document.querySelectorAll('.conference-tab').forEach(t => {
                    t.classList.remove('active');
                    t.style.background = '#f1f5f9';
                    t.style.color = '#64748b';
                    t.style.border = '2px solid #e2e8f0';
                });
                e.target.classList.add('active');
                e.target.style.background = '#ea580c';
                e.target.style.color = 'white';
                e.target.style.border = '2px solid #ea580c';
                
                const conference = e.target.dataset.conference;
                const divisions = this.data.team_conferences[conference + ' Conference'];
                this.renderConferenceDivisions(
                    document.getElementById('team-divisions-container'), 
                    divisions, 
                    conference
                );
                this.bindTeamSelectionListeners();
            });
        });

        // Initial team selection binding
        this.bindTeamSelectionListeners();

        // Player search
        const playerInput = document.getElementById('player-search-input');
        if (playerInput) {
            playerInput.addEventListener('input', (e) => {
                this.handlePlayerSearch(e.target.value);
            });
        }

        // Top shooters selection will be bound after UI is populated

        // Era buttons
        document.querySelectorAll('.era-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.era-btn').forEach(b => {
                    b.classList.remove('active');
                    b.style.background = '#f8fafc';
                    b.style.color = '#475569';
                    b.style.border = '2px solid #e2e8f0';
                });
                e.target.classList.add('active');
                e.target.style.background = '#ea580c';
                e.target.style.color = 'white';
                e.target.style.border = '2px solid #ea580c';
                
                const [start, end] = e.target.dataset.era.split('-').map(Number);
                this.timeRange = { start, end };
                
                document.getElementById('start-year-slider').value = start;
                document.getElementById('end-year-slider').value = end;
                document.getElementById('start-year-display').textContent = start;
                document.getElementById('end-year-display').textContent = end;
                
                this.updateVisualization();
            });
        });

        // Year sliders
        ['start', 'end'].forEach(type => {
            const slider = document.getElementById(`${type}-year-slider`);
            const display = document.getElementById(`${type}-year-display`);
            
            if (slider && display) {
                slider.addEventListener('input', (e) => {
                    const value = parseInt(e.target.value);
                    display.textContent = value;
                    this.timeRange[type] = value;
                    this.updateVisualization();
                });
            }
        });
    },

    bindTeamSelectionListeners() {
        document.querySelectorAll('.team-option').forEach(option => {
            option.addEventListener('click', (e) => {
                const team = e.target.dataset.team;
                
                if (this.selectedTeams.has(team)) {
                    this.selectedTeams.delete(team);
                    e.target.classList.remove('selected');
                    e.target.style.background = 'white';
                    e.target.style.color = 'black';
                    e.target.style.border = '2px solid transparent';
                } else {
                    this.selectedTeams.add(team);
                    e.target.classList.add('selected');
                    e.target.style.background = '#ea580c';
                    e.target.style.color = 'white';
                    e.target.style.border = '2px solid #ea580c';
                }
                
                this.updateSelectedCount();
                this.updateVisualization();
            });
        });
    },

    bindShooterSelectionListeners() {
        // Bind shooters section player cards
        const playerCards = document.querySelectorAll('.player-card');
        console.log('Binding shooters section player cards:', playerCards.length);
        playerCards.forEach(card => {
            card.addEventListener('click', (e) => {
                const player = e.currentTarget.dataset.player;
                console.log('Player card clicked:', player);
                this.togglePlayerSelection(player, e.currentTarget, true);
            });
        });
        
        // Bind player list options  
        const playerListOptions = document.querySelectorAll('.player-list-option');
        console.log('Binding player list options:', playerListOptions.length);
        playerListOptions.forEach(option => {
            option.addEventListener('click', (e) => {
                const player = e.currentTarget.dataset.player;
                console.log('Player list option clicked:', player);
                this.togglePlayerSelection(player, e.currentTarget, false);
            });
        });
    },

    togglePlayerSelection(player, element, isShooterCard) {
        console.log(`🏀 Toggle player selection: ${player}, currently selected:`, Array.from(this.selectedPlayers));
        
        if (this.selectedPlayers.has(player)) {
            this.selectedPlayers.delete(player);
            element.classList.remove('selected');
            if (isShooterCard) {
                element.style.background = '#f8fafc';
                element.style.color = 'black';
            } else {
                element.style.background = 'white';
                element.style.color = 'black';
                element.style.border = '2px solid #e2e8f0';
            }
            console.log(`📤 Removed ${player}, now selected:`, Array.from(this.selectedPlayers));
        } else {
            this.selectedPlayers.add(player);
            element.classList.add('selected');
            if (isShooterCard) {
                element.style.background = 'linear-gradient(135deg, #ea580c, #dc2626)';
                element.style.color = 'white';
            } else {
                element.style.background = '#ea580c';
                element.style.color = 'white';
                element.style.border = '2px solid #ea580c';
            }
            console.log(`📥 Added ${player}, now selected:`, Array.from(this.selectedPlayers));
        }
        
        // Update both displays to keep them in sync
        this.syncPlayerSelections();
        this.updateSelectedPlayersDisplay();
        console.log(`🔄 Calling updateVisualization for player selection change...`);
        this.updateVisualization();
    },

    syncPlayerSelections() {
        // Sync shooters section
        document.querySelectorAll('.player-card').forEach(card => {
            const player = card.dataset.player;
            if (this.selectedPlayers.has(player)) {
                card.classList.add('selected');
                card.style.background = 'linear-gradient(135deg, #ea580c, #dc2626)';
                card.style.color = 'white';
            } else {
                card.classList.remove('selected');
                card.style.background = '#f8fafc';
                card.style.color = 'black';
            }
        });
        
        // Sync player list
        document.querySelectorAll('.player-list-option').forEach(option => {
            const player = option.dataset.player;
            if (this.selectedPlayers.has(player)) {
                option.classList.add('selected');
                option.style.background = '#ea580c';
                option.style.color = 'white';
                option.style.border = '2px solid #ea580c';
            } else {
                option.classList.remove('selected');
                option.style.background = 'white';
                option.style.color = 'black';
                option.style.border = '2px solid #e2e8f0';
            }
        });
    },

    updateViewSections() {
        const sections = {
            'team-filter-section': this.currentView === 'teams',
            'player-filter-section': this.currentView === 'players',
            'shooters-section': this.currentView === 'shooters'
        };

        Object.entries(sections).forEach(([id, show]) => {
            const element = document.getElementById(id);
            if (element) {
                element.style.display = show ? 'block' : 'none';
            }
        });

        // Update chart title
        const titles = {
            'teams': 'Team Three-Point Trends',
            'players': 'Player Comparison Analysis',
            'shooters': 'Elite Shooter Performance'
        };
        
        document.getElementById('advanced-chart-title').textContent = titles[this.currentView];
        
        // Update chart option buttons based on view
        this.updateChartOptionButtons();
    },

    updateChartOptionButtons() {
        const container = document.getElementById('chart-option-buttons');
        if (!container) return;

        // Define buttons for each view type
        const buttonConfigs = {
            'teams': [
                { view: 'trends', label: '📈 Trends', active: true }
            ],
            'players': [
                { view: 'trends', label: '📈 Trends', active: true }
            ],
            'shooters': [
                { view: 'trends', label: '📈 Rankings', active: true }
            ]
        };

        const buttons = buttonConfigs[this.currentView] || [];
        
        container.innerHTML = buttons.map(btn => `
            <button class="chart-option-btn ${btn.active ? 'active' : ''}" 
                    data-view="${btn.view}" 
                    style="padding: 6px 12px; background: ${btn.active ? '#ea580c' : '#f1f5f9'}; color: ${btn.active ? 'white' : '#64748b'}; border: 1px solid #e2e8f0; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 500;">
                ${btn.label}
            </button>
        `).join('');

        // Rebind button event listeners
        this.bindChartOptionButtons();
    },

    bindChartOptionButtons() {
        document.querySelectorAll('.chart-option-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.chart-option-btn').forEach(b => {
                    b.classList.remove('active');
                    b.style.background = '#f1f5f9';
                    b.style.color = '#64748b';
                });
                e.target.classList.add('active');
                e.target.style.background = '#ea580c';
                e.target.style.color = 'white';
                this.updateVisualization();
            });
        });
    },

    updateSelectedCount() {
        const count = this.currentView === 'teams' ? this.selectedTeams.size : this.selectedPlayers.size;
        const label = this.currentView === 'teams' ? 'Teams Selected' : 'Players Selected';
        
        document.getElementById('selected-count').textContent = count;
        document.querySelector('#selected-count').nextElementSibling.textContent = label;
    },

    updateSelectedPlayersDisplay() {
        const container = document.getElementById('selected-players-list');
        if (!container) return;

        container.innerHTML = Array.from(this.selectedPlayers).map(player => `
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: linear-gradient(135deg, #ea580c, #dc2626); color: white; border-radius: 6px; font-size: 12px; font-weight: 500;">
                <span>${player}</span>
                <button class="remove-player-btn" data-player="${player}" style="background: rgba(255, 255, 255, 0.2); color: white; border: none; border-radius: 50%; width: 20px; height: 20px; cursor: pointer; font-size: 14px; line-height: 1;">×</button>
            </div>
        `).join('');

        // Bind remove buttons
        container.querySelectorAll('.remove-player-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const player = e.target.dataset.player;
                this.selectedPlayers.delete(player);
                this.updateSelectedPlayersDisplay();
                this.updateVisualization();
                
                // Update top shooters list
                document.querySelectorAll('.player-card').forEach(card => {
                    if (card.dataset.player === player) {
                        card.classList.remove('selected');
                        card.style.background = '#f8fafc';
                        card.style.color = 'black';
                    }
                });
            });
        });
    },

    handlePlayerSearch(query) {
        const suggestions = document.getElementById('player-suggestions');
        if (!suggestions) return;

        if (query.length < 2) {
            suggestions.style.display = 'none';
            return;
        }

        const matchingPlayers = this.data.player_data
            .filter(p => p.player.toLowerCase().includes(query.toLowerCase()))
            .slice(0, 10);

        if (matchingPlayers.length > 0) {
            suggestions.innerHTML = matchingPlayers.map(player => `
                <div style="padding: 8px 12px; cursor: pointer; border-bottom: 1px solid #e2e8f0; transition: background-color 0.2s ease;" 
                     onmouseover="this.style.background='#f1f5f9'" 
                     onmouseout="this.style.background='white'"
                     onclick="window.nbaAdvancedExplorer.selectPlayerFromSearch('${player.player}')">
                    ${player.player}
                </div>
            `).join('');
            suggestions.style.display = 'block';
        } else {
            suggestions.style.display = 'none';
        }
    },

    selectPlayerFromSearch(playerName) {
        this.selectedPlayers.add(playerName);
        this.updateSelectedPlayersDisplay();
        this.updateVisualization();
        
        document.getElementById('player-search-input').value = '';
        document.getElementById('player-suggestions').style.display = 'none';
    },

    showError(message) {
        const container = document.querySelector('.advanced-visualization-area') || 
                         document.getElementById('explorer-dashboard');
        container.innerHTML = `
            <div style="background: #fee2e2; color: #991b1b; padding: 20px; border-radius: 8px; border: 1px solid #fca5a5; text-align: center;">
                <h2 style="margin: 0 0 12px 0; font-size: 18px;">Error</h2>
                <p style="margin: 0 0 16px 0; font-size: 14px;">${message}</p>
                <button onclick="location.reload()" style="background: #dc2626; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 500;">Refresh Page</button>
            </div>
        `;
    }
});

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { NBAAdvancedExplorer };
}// Enhanced NBA Explorer - Visualization Functions

// Add visualization methods to the NBAAdvancedExplorer class
Object.assign(NBAAdvancedExplorer.prototype, {
    
    updateVisualization() {
        console.log(`📈 updateVisualization called with currentView: ${this.currentView}`);
        
        const svg = d3.select('#advanced-main-chart');
        svg.selectAll('*').remove();

        const container = svg.node().parentElement;
        const width = container.clientWidth - 40;
        const height = container.clientHeight - 40;

        svg.attr('width', width).attr('height', height);

        const margin = { top: 20, right: 30, bottom: 40, left: 60 };
        const chartWidth = width - margin.left - margin.right;
        const chartHeight = height - margin.top - margin.bottom;

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        switch (this.currentView) {
            case 'teams':
                console.log('📊 Drawing team trends');
                this.drawTeamTrends(g, chartWidth, chartHeight);
                break;
            case 'players':
                console.log('📊 Drawing player comparison');
                this.drawPlayerComparison(g, chartWidth, chartHeight);
                break;
            case 'shooters':
                console.log('📊 Drawing shooter analysis');
                this.drawShooterAnalysis(g, chartWidth, chartHeight);
                break;
            default:
                console.log(`⚠️ Unknown view: ${this.currentView}`);
        }

        this.updateInsights();
        this.updateQuickStats();
    },

    setCurrentView(viewType) {
        // Map external view types to internal view types
        const viewMapping = {
            'players': 'players',
            'teams': 'teams'
        };
        
        const newView = viewMapping[viewType] || 'teams';
        if (this.currentView !== newView) {
            this.currentView = newView;
            console.log(`🔄 Enhanced explorer view switched to: ${newView}`);
            this.updateViewSections();
            
            // Re-bind listeners when switching to players view
            if (newView === 'players') {
        
                this.bindShooterSelectionListeners();
            } else if (newView === 'teams') {
        
                this.bindTeamSelectionListeners();
            }
            
            this.updateVisualization();
        }
    },

    drawTeamTrends(g, width, height) {
        if (this.selectedTeams.size === 0) {
            g.append('text')
                .attr('x', width / 2)
                .attr('y', height / 2)
                .attr('text-anchor', 'middle')
                .style('font-size', '16px')
                .style('fill', '#64748b')
                .text('Select teams from the left panel to view trends');
            return;
        }

        // Enhanced debugging
        console.log('🏟️ Selected teams:', Array.from(this.selectedTeams));
        console.log('🏟️ Available team data structure:', this.data.team_data?.length || 0, 'teams');
        if (this.data.team_data && this.data.team_data.length > 0) {
            console.log('🏟️ First team structure:', this.data.team_data[0]);
            console.log('🏟️ All available team names:', this.data.team_data.map(t => t.team).slice(0, 10));
        }
        
        const selectedTeamData = this.data.team_data.filter(team => 
            this.selectedTeams.has(team.team)
        );
        
        console.log('🏟️ Found team data for:', selectedTeamData.map(t => t.team));
        console.log('🏟️ Selected team data count:', selectedTeamData.length);

        const allSeasons = [];
        selectedTeamData.forEach(team => {
            team.seasons.forEach(season => {
                // Fix: use season.season instead of season.year
                if (season.season >= this.timeRange.start && season.season <= this.timeRange.end) {
                    allSeasons.push({
                        ...season,
                        team: team.team,
                        year: season.season // Add year property for consistency
                    });
                }
            });
        });
        
        console.log('🏟️ All seasons data:', allSeasons.length, 'seasons found');
        console.log('🏟️ Time range:', this.timeRange.start, 'to', this.timeRange.end);

        if (allSeasons.length === 0) {
            g.append('text')
                .attr('x', width / 2)
                .attr('y', height / 2)
                .attr('text-anchor', 'middle')
                .style('font-size', '16px')
                .style('fill', '#64748b')
                .text('No data available for selected time range');
            return;
        }

        // Scales
        const xScale = d3.scaleLinear()
            .domain([this.timeRange.start, this.timeRange.end])
            .range([0, width]);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(allSeasons, d => d.three_pt_shots)])
            .range([height, 0]);

        const colorScale = d3.scaleOrdinal()
            .domain(Array.from(this.selectedTeams))
            .range(['#ea580c', '#1e40af', '#16a34a', '#f59e0b', '#dc2626', '#7c3aed', '#0891b2', '#be185d']);

        // Add grid
        g.append('g')
            .attr('class', 'grid')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(xScale)
                .tickSize(-height)
                .tickFormat('')
            )
            .style('stroke-dasharray', '2,2')
            .style('opacity', 0.3);

        g.append('g')
            .attr('class', 'grid')
            .call(d3.axisLeft(yScale)
                .tickSize(-width)
                .tickFormat('')
            )
            .style('stroke-dasharray', '2,2')
            .style('opacity', 0.3);

        // Axes
        g.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(xScale).tickFormat(d3.format('d')))
            .style('font-size', '12px');

        g.append('g')
            .call(d3.axisLeft(yScale).tickFormat(d => d.toLocaleString()))
            .style('font-size', '12px');

        // Lines for each team
        const line = d3.line()
            .x(d => xScale(d.season))
                            .y(d => yScale(d.three_pt_shots))
            .curve(d3.curveMonotoneX);

        selectedTeamData.forEach((team, index) => {
            const teamSeasons = team.seasons.filter(s => 
                s.season >= this.timeRange.start && s.season <= this.timeRange.end
            );

            if (teamSeasons.length === 0) return;

            // Draw line
            g.append('path')
                .datum(teamSeasons)
                .attr('fill', 'none')
                .attr('stroke', colorScale(team.team))
                .attr('stroke-width', 3)
                .attr('d', line)
                .style('opacity', 0.8);

            // Add points
            g.selectAll(`.points-${index}`)
                .data(teamSeasons)
                .enter()
                .append('circle')
                .attr('cx', d => xScale(d.season))
                .attr('cy', d => yScale(d.three_pt_shots))
                .attr('r', 4)
                .attr('fill', colorScale(team.team))
                .attr('stroke', 'white')
                .attr('stroke-width', 2)
                .style('cursor', 'pointer')
                .on('mouseover', (event, d) => {
                    this.showAdvancedTooltip(event, `
                        <strong>${team.team} - ${d.season}</strong><br/>
                        3PT Attempts: ${d.three_pt_shots?.toLocaleString() || 'N/A'}<br/>
                        3PT Made: ${d.three_pt_made?.toLocaleString() || 'N/A'}<br/>
                        3PT %: ${d.three_pt_percentage || 'N/A'}%
                    `);
                })
                .on('mouseout', this.hideAdvancedTooltip);
        });

        // Legend
        const legend = g.append('g')
            .attr('transform', `translate(${width - 150}, 20)`);

        selectedTeamData.forEach((team, index) => {
            const legendItem = legend.append('g')
                .attr('transform', `translate(0, ${index * 20})`);

            legendItem.append('line')
                .attr('x1', 0)
                .attr('x2', 15)
                .attr('y1', 0)
                .attr('y2', 0)
                .attr('stroke', colorScale(team.team))
                .attr('stroke-width', 3);

            legendItem.append('text')
                .attr('x', 20)
                .attr('y', 4)
                .style('font-size', '11px')
                .style('font-weight', '500')
                .text(team.team.length > 15 ? team.team.substring(0, 15) + '...' : team.team);
        });

        // Axis labels
        g.append('text')
            .attr('x', width / 2)
            .attr('y', height + 35)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px')
            .style('font-weight', '500')
            .text('Season');

        g.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('x', -height / 2)
            .attr('y', -40)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px')
            .style('font-weight', '500')
            .text('Three-Point Attempts');
    },

    drawPlayerComparison(g, width, height) {
        console.log(`📊 drawPlayerComparison called with ${this.selectedPlayers.size} selected players:`, Array.from(this.selectedPlayers));
        
        if (this.selectedPlayers.size === 0) {
            console.log('📋 No players selected, showing instruction text');
            g.append('text')
                .attr('x', width / 2)
                .attr('y', height / 2)
                .attr('text-anchor', 'middle')
                .style('font-size', '16px')
                .style('fill', '#64748b')
                .text('Select players to compare their three-point evolution');
            return;
        }

        // Debug: Check selected players and available player data
        console.log('Selected players:', Array.from(this.selectedPlayers));
        console.log('Available top shooters:', this.topShooters ? this.topShooters.map(p => p.player) : 'No topShooters');
        
        // Use topShooters data directly since it has the correct structure with seasons_data
        const selectedPlayerData = Array.from(this.selectedPlayers).map(playerName => {
            return this.topShooters ? this.topShooters.find(p => p.player === playerName) : null;
        }).filter(Boolean);
        
        console.log('Found player data for:', selectedPlayerData.map(p => p.player));

        if (selectedPlayerData.length === 0) {
            g.append('text')
                .attr('x', width / 2)
                .attr('y', height / 2)
                .attr('text-anchor', 'middle')
                .style('font-size', '16px')
                .style('fill', '#64748b')
                .text('Selected players not found in dataset');
            return;
        }

        const allSeasons = [];
        selectedPlayerData.forEach(player => {
            // Use seasons_data from topShooters instead of seasons from master data
            const playerSeasons = player.seasons_data || player.seasons || [];
            playerSeasons.forEach(season => {
                if (season.year >= this.timeRange.start && season.year <= this.timeRange.end) {
                    allSeasons.push({
                        ...season,
                        player: player.player
                    });
                }
            });
        });

        if (allSeasons.length === 0) return;

        // Scales
        const xScale = d3.scaleLinear()
            .domain([this.timeRange.start, this.timeRange.end])
            .range([0, width]);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(allSeasons, d => d.three_pt_percentage || 0)])
            .range([height, 0]);

        const colorScale = d3.scaleOrdinal()
            .domain(Array.from(this.selectedPlayers))
            .range(['#ea580c', '#1e40af', '#16a34a', '#f59e0b', '#dc2626', '#7c3aed', '#0891b2', '#be185d']);

        // Add grid
        g.append('g')
            .attr('class', 'grid')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(xScale).tickSize(-height).tickFormat(''))
            .style('stroke-dasharray', '2,2')
            .style('opacity', 0.3);

        g.append('g')
            .attr('class', 'grid')
            .call(d3.axisLeft(yScale).tickSize(-width).tickFormat(''))
            .style('stroke-dasharray', '2,2')
            .style('opacity', 0.3);

        // Axes
        g.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(xScale).tickFormat(d3.format('d')));

        g.append('g')
            .call(d3.axisLeft(yScale).tickFormat(d => d + '%'));

        // Lines for each player
        const line = d3.line()
            .x(d => xScale(d.year))
            .y(d => yScale(d.three_pt_percentage || 0))
            .curve(d3.curveMonotoneX);

        selectedPlayerData.forEach((player, index) => {
            // Use seasons_data from topShooters instead of seasons from master data
            const allPlayerSeasons = player.seasons_data || player.seasons || [];
            const playerSeasons = allPlayerSeasons.filter(s => 
                s.year >= this.timeRange.start && s.year <= this.timeRange.end
            );

            if (playerSeasons.length === 0) return;

            // Draw line
            g.append('path')
                .datum(playerSeasons)
                .attr('fill', 'none')
                .attr('stroke', colorScale(player.player))
                .attr('stroke-width', 3)
                .attr('d', line)
                .style('opacity', 0.8);

            // Add points
            g.selectAll(`.player-points-${index}`)
                .data(playerSeasons)
                .enter()
                .append('circle')
                .attr('cx', d => xScale(d.year))
                .attr('cy', d => yScale(d.three_pt_percentage || 0))
                .attr('r', 4)
                .attr('fill', colorScale(player.player))
                .attr('stroke', 'white')
                .attr('stroke-width', 2)
                .style('cursor', 'pointer')
                .on('mouseover', (event, d) => {
                    this.showAdvancedTooltip(event, `
                        <strong>${player.player} - ${d.year}</strong><br/>
                        3PT%: ${d.three_pt_percentage || 'N/A'}%<br/>
                        Made/Attempts: ${d.three_pt_made || 'N/A'}/${d.three_pt_attempts || 'N/A'}<br/>
                        Team: ${d.team || 'N/A'}
                    `);
                })
                .on('mouseout', this.hideAdvancedTooltip);
        });

        // Legend
        const legend = g.append('g')
            .attr('transform', `translate(${width - 150}, 20)`);

        selectedPlayerData.forEach((player, index) => {
            const legendItem = legend.append('g')
                .attr('transform', `translate(0, ${index * 20})`);

            legendItem.append('line')
                .attr('x1', 0)
                .attr('x2', 15)
                .attr('y1', 0)
                .attr('y2', 0)
                .attr('stroke', colorScale(player.player))
                .attr('stroke-width', 3);

            legendItem.append('text')
                .attr('x', 20)
                .attr('y', 4)
                .style('font-size', '11px')
                .style('font-weight', '500')
                .text(player.player.length > 15 ? player.player.substring(0, 15) + '...' : player.player);
        });

        // Axis labels
        g.append('text')
            .attr('x', width / 2)
            .attr('y', height + 35)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px')
            .style('font-weight', '500')
            .text('Season');

        g.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('x', -height / 2)
            .attr('y', -40)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px')
            .style('font-weight', '500')
            .text('Three-Point Percentage');
    },

    drawShooterAnalysis(g, width, height) {
        const displayShooters = this.selectedPlayers.size > 0 ? 
            this.topShooters.filter(s => this.selectedPlayers.has(s.player)) :
            this.topShooters.slice(0, 10);

        if (displayShooters.length === 0) {
            g.append('text')
                .attr('x', width / 2)
                .attr('y', height / 2)
                .attr('text-anchor', 'middle')
                .style('font-size', '16px')
                .style('fill', '#64748b')
                .text('Top 10 three-point shooters shown by default');
            return;
        }

        const xScale = d3.scaleBand()
            .domain(displayShooters.map(d => d.player))
            .range([0, width])
            .padding(0.2);

        const yScale = d3.scaleLinear()
            .domain([0, 50])
            .range([height, 0]);

        // Bars
        g.selectAll('.shooter-bar')
            .data(displayShooters)
            .enter()
            .append('rect')
            .attr('class', 'shooter-bar')
            .attr('x', d => xScale(d.player))
            .attr('y', d => yScale(d.career_three_pt_percentage))
            .attr('width', xScale.bandwidth())
            .attr('height', d => height - yScale(d.career_three_pt_percentage))
            .attr('fill', this.colors.primary)
            .attr('rx', 3)
            .style('opacity', 0.8)
            .style('cursor', 'pointer')
            .on('mouseover', (event, d) => {
                this.showAdvancedTooltip(event, `
                    <strong>${d.player}</strong><br/>
                    Career 3PT%: ${d.career_three_pt_percentage}%<br/>
                    Made/Attempts: ${d.career_three_pt_made}/${d.career_three_pt_attempts}<br/>
                    Active: ${d.years_active}<br/>
                    Seasons: ${d.seasons_played}
                `);
            })
            .on('mouseout', this.hideAdvancedTooltip);

        // Percentage labels on bars
        g.selectAll('.percentage-label')
            .data(displayShooters)
            .enter()
            .append('text')
            .attr('class', 'percentage-label')
            .attr('x', d => xScale(d.player) + xScale.bandwidth() / 2)
            .attr('y', d => yScale(d.career_three_pt_percentage) - 5)
            .attr('text-anchor', 'middle')
            .style('font-size', '11px')
            .style('font-weight', '600')
            .style('fill', this.colors.primary)
            .text(d => d.career_three_pt_percentage + '%');

        // Axes
        g.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(xScale))
            .selectAll('text')
            .style('font-size', '10px')
            .attr('transform', 'rotate(-45)')
            .style('text-anchor', 'end');

        g.append('g')
            .call(d3.axisLeft(yScale).tickFormat(d => d + '%'));

        // Axis labels
        g.append('text')
            .attr('x', width / 2)
            .attr('y', height + 35)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px')
            .style('font-weight', '500')
            .text('Players');

        g.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('x', -height / 2)
            .attr('y', -40)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px')
            .style('font-weight', '500')
            .text('Career Three-Point Percentage');
    },

    updateInsights() {
        const container = document.getElementById('chart-insights');
        const insights = this.generateInsights();
        
        container.innerHTML = insights.map(insight => `
            <div style="display: flex; align-items: center; gap: 12px; padding: 8px 0; border-bottom: 1px solid #e2e8f0;">
                <div style="font-size: 16px; width: 20px; text-align: center;">${insight.icon}</div>
                <div style="font-size: 13px; color: #374151; line-height: 1.4;">${insight.text}</div>
            </div>
        `).join('');
    },

    updateQuickStats() {
        const container = document.getElementById('quick-stats-content');
        const stats = this.generateQuickStats();
        
        container.innerHTML = stats.map(stat => `
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: #f8fafc; border-radius: 6px; margin-bottom: 6px; border-left: 3px solid #ea580c;">
                <div style="font-size: 11px; color: #64748b; font-weight: 500;">${stat.label}</div>
                <div style="font-size: 14px; font-weight: 700; color: #ea580c;">${stat.value}</div>
            </div>
        `).join('');
    },

    generateInsights() {
        const insights = [];
        
        if (this.currentView === 'teams' && this.selectedTeams.size > 0) {
            insights.push({
                icon: '📈',
                text: `Analyzing ${this.selectedTeams.size} teams over ${this.timeRange.end - this.timeRange.start + 1} seasons`
            });
        }
        
        if (this.selectedPlayers.size > 0) {
            insights.push({
                icon: '🏀',
                text: `Comparing ${this.selectedPlayers.size} elite shooters from the analysis`
            });
        }
        
        insights.push({
            icon: '🎯',
            text: `Three-point attempts have increased by over 140% since 2004`
        });
        
        insights.push({
            icon: '📊',
            text: `Use the time range controls to focus on specific eras of basketball evolution`
        });
        
        return insights;
    },

    generateQuickStats() {
        const stats = [];
        
        if (this.currentView === 'teams') {
            stats.push({
                label: 'Teams Selected',
                value: this.selectedTeams.size
            });
            stats.push({
                label: 'Years Analyzed',
                value: this.timeRange.end - this.timeRange.start + 1
            });
        } else if (this.currentView === 'players') {
            stats.push({
                label: 'Players Selected',
                value: this.selectedPlayers.size
            });
        } else if (this.currentView === 'shooters') {
            const selected = this.selectedPlayers.size;
            const showing = selected > 0 ? selected : 10;
            stats.push({
                label: 'Shooters Shown',
                value: showing
            });
            
            if (this.topShooters.length > 0) {
                stats.push({
                    label: 'Top Shooter',
                    value: this.topShooters[0].career_three_pt_percentage + '%'
                });
            }
        }
        
        return stats;
    },

    showAdvancedTooltip(event, content) {
        const tooltip = d3.select('body').selectAll('.advanced-tooltip')
            .data([0])
            .join('div')
            .attr('class', 'advanced-tooltip')
            .style('position', 'absolute')
            .style('background', '#0f172a')
            .style('color', 'white')
            .style('padding', '8px 12px')
            .style('border-radius', '6px')
            .style('font-size', '12px')
            .style('font-weight', '500')
            .style('pointer-events', 'none')
            .style('z-index', '10000')
            .style('max-width', '250px')
            .style('box-shadow', '0 4px 20px rgba(0, 0, 0, 0.3)')
            .style('opacity', 0);

        tooltip.html(content)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .transition()
            .duration(200)
            .style('opacity', 1);
    },

    hideAdvancedTooltip() {
        d3.select('.advanced-tooltip')
            .transition()
            .duration(200)
            .style('opacity', 0)
            .remove();
    }
});

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { NBAAdvancedExplorer };
}