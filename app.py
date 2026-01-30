"""
Valorant Team Analytics Dashboard

A comprehensive analytics dashboard for Valorant scrim data.
Processes JSON files from OCR tools and provides detailed team and player statistics.
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


# ============================================================================
# CONFIGURATION - Easy to edit formulas and constants
# ============================================================================

class Config:
    """Central configuration for all constants and formulas."""
    
    # Validation constants
    MIN_VALID_ROUNDS = 13
    MAX_VALID_ROUNDS = 999  # No upper limit
    
    # Time windows (seconds)
    TRADE_WINDOW_SECONDS = 5
    ROUND_MAX_DURATION = 150
    
    # Economy thresholds
    ECO_ROUND_THRESHOLD = 2000
    FULL_BUY_THRESHOLD = 4000
    
    # Rating formula weights (must sum to 1.0)
    RATING_WEIGHTS = {
        'kpr': 0.30,              # Kill participation
        'fb_impact': 0.20,        # First blood impact (FBPR * 1.5)
        'kast': 0.20,             # Consistency
        'win_contribution': 0.15, # Round win rate
        'teamplay': 0.10,         # Trade rate
        'death_penalty': 0.05,    # Deaths (negative)
    }
    
    # First blood multiplier for rating
    FB_IMPACT_MULTIPLIER = 1.5
    
    # Stats display precision
    STAT_PRECISION = {
        'kd': 2,
        'kpr': 3,
        'dpr': 3,
        'fbpr': 3,
        'fdpr': 3,
        'percentage': 1,
        'rating': 2,
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_player_name(name: str) -> str:
    """Clean and normalize player names for consistency.
    
    Handles OCR errors and standardizes formatting.
    
    Args:
        name: Raw player name from OCR
        
    Returns:
        Cleaned, normalized player name
    """
    if not name:
        return ""
    
    # Known OCR mistakes - add new ones here as you find them
    name_aliases = {
        "snx|jaywonning4z": "snx|jaywonning42067",
        "snx|jaywonning4z067": "snx|jaywonning42067",
    }
    
    # Basic cleaning
    cleaned = name.strip().lower().replace(" ", "")
    
    # Apply known aliases
    return name_aliases.get(cleaned, cleaned)


def safe_percentage(numerator: float, denominator: float) -> float:
    """Calculate percentage with zero-division protection.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        
    Returns:
        Percentage value or 0.0 if denominator is 0
    """
    return (numerator / denominator * 100) if denominator > 0 else 0.0


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide with zero-division protection.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Value to return if denominator is 0
        
    Returns:
        Division result or default if denominator is 0
    """
    return (numerator / denominator) if denominator > 0 else default


# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4655;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0F1923;
        margin-top: 1.5rem;
    }
    .metric-card {
        background-color: #1A1F2C;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF4655;
    }
    .positive {
        color: #00FF00;
    }
    .negative {
        color: #FF4655;
    }
</style>
"""


# ============================================================================
# MAIN ANALYZER CLASS
# ============================================================================

class ValorantAnalyzer:
    """Main class for analyzing Valorant match data."""
    
    def __init__(self, data_folder: str = "."):
        """Initialize the analyzer.
        
        Args:
            data_folder: Path to folder containing JSON match files
        """
        self.data_folder = data_folder
        self.matches = []
        self.allowed_players = set()
        self.players_file_exists = False
        self._load_allowed_players()
        self._load_all_matches()
    
    def _load_allowed_players(self) -> None:
        """Load allowed players from players.txt."""
        players_file = os.path.join(self.data_folder, "players.txt")
        
        if not os.path.exists(players_file):
            st.sidebar.warning("⚠️ players.txt not found. Will use all players from matches.")
            return
        
        self.players_file_exists = True
        
        try:
            with open(players_file, encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                
            if not lines:
                st.sidebar.warning("⚠️ players.txt is empty. Will use all players from matches.")
                return
            
            for player in lines:
                self.allowed_players.add(clean_player_name(player))
            
            st.sidebar.success(f"✓ Loaded {len(self.allowed_players)} players from players.txt")
            
        except Exception as e:
            st.error(f"❌ Error reading players.txt: {e}")
    
    def _load_all_matches(self) -> None:
    """Load all JSON match files from data folder."""
    
    self.matches = []

    # Make path absolute based on app.py location
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, self.data_folder)

    if not os.path.exists(data_path):
        st.warning(f"⚠️ Data folder '{data_path}' not found.")
        return

    json_files = [f for f in os.listdir(data_path) if f.endswith(".json")]

    loaded = 0
    failed = 0

    for filename in json_files:
        filepath = os.path.join(data_path, filename)
        try:
            with open(filepath, encoding="utf-8") as f:
                match_data = json.load(f)
            
            if not self._validate_match_data(match_data, filename):
                failed += 1
                continue
            
            match_data['filename'] = filename
            
            try:
                match_data['parsed_date'] = datetime.strptime(match_data['date'], '%d/%m/%Y')
            except Exception:
                st.warning(f"⚠️ Invalid date format in {filename}, using current date")
                match_data['parsed_date'] = datetime.now()
            
            self.matches.append(match_data)
            loaded += 1

        except json.JSONDecodeError:
            st.error(f"❌ Invalid JSON in {filename}")
            failed += 1
        except Exception as e:
            st.error(f"❌ Error loading {filename}: {e}")
            failed += 1

    self.matches.sort(key=lambda x: x['parsed_date'])

    if loaded > 0:
        st.sidebar.info(f"📊 Loaded {loaded} matches ({failed} failed)")



    def _validate_match_data(self, data: Dict, filename: str) -> bool:
        """Validate match data structure.
        
        Args:
            data: Match data dictionary
            filename: Name of file being validated
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['players_agents', 'rounds', 'map_name']
        missing = [f for f in required_fields if f not in data]
        
        if missing:
            st.warning(f"⚠️ {filename} missing fields: {missing}")
            return False
        
        if not isinstance(data['rounds'], list):
            st.warning(f"⚠️ {filename}: 'rounds' must be a list")
            return False
        
        num_rounds = len(data['rounds'])
        if num_rounds < Config.MIN_VALID_ROUNDS:
            st.warning(f"⚠️ {filename}: Only {num_rounds} rounds (minimum {Config.MIN_VALID_ROUNDS})")
            return False
        
        return True
    
    def get_all_players(self) -> List[str]:
        """Get list of all unique players.
        
        Returns only allowed players if players.txt exists and has content,
        otherwise returns all players from matches.
        
        Returns:
            Sorted list of player names (cleaned)
        """
        players_in_matches = set()
        
        if self.players_file_exists and self.allowed_players:
            # Return only allowed players that appear in matches
            for match in self.matches:
                for player in match['players_agents'].keys():
                    if clean_player_name(player) in self.allowed_players:
                        players_in_matches.add(clean_player_name(player))
            return sorted(list(players_in_matches))
        else:
            # Return all players (cleaned names)
            for match in self.matches:
                for player in match['players_agents'].keys():
                    players_in_matches.add(clean_player_name(player))
            return sorted(list(players_in_matches))
    
    def get_all_maps(self) -> List[str]:
        """Get list of all unique maps.
        
        Returns:
            Sorted list of map names
        """
        maps = {match['map_name'] for match in self.matches}
        return sorted(list(maps))
    
    def filter_matches(
        self,
        selected_maps: Optional[List[str]] = None,
        selected_players: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        last_n_matches: Optional[int] = None
    ) -> List[Dict]:
        """Filter matches based on criteria.
        
        Args:
            selected_maps: List of maps to include
            selected_players: List of players to include (cleaned names)
            start_date: Minimum date
            end_date: Maximum date
            last_n_matches: Number of most recent matches
            
        Returns:
            Filtered list of matches
        """
        filtered = self.matches.copy()
        
        # Date filter
        if start_date:
            filtered = [m for m in filtered if m['parsed_date'] >= start_date]
        if end_date:
            filtered = [m for m in filtered if m['parsed_date'] <= end_date]
        
        # Last N matches
        if last_n_matches and last_n_matches > 0:
            filtered = filtered[-last_n_matches:]
        
        # Map filter
        if selected_maps:
            filtered = [m for m in filtered if m['map_name'] in selected_maps]
        
        # Player filter (match must include at least one selected player)
        if selected_players:
            filtered = [
                m for m in filtered
                if any(clean_player_name(p) in selected_players for p in m['players_agents'].keys())
            ]
        
        return filtered
    
    def is_allowed_player(self, player_name: str) -> bool:
        """Check if a player is in the allowed list.
        
        Args:
            player_name: Player name to check
            
        Returns:
            True if player is allowed or no filter exists
        """
        if not self.players_file_exists or not self.allowed_players:
            return True
        cleaned = clean_player_name(player_name)
        return cleaned in self.allowed_players
    
    def calculate_player_stats(self, filtered_matches: List[Dict]) -> pd.DataFrame:
        """Calculate comprehensive player statistics.
        
        Args:
            filtered_matches: List of matches to analyze
            
        Returns:
            DataFrame with player statistics
        """
        # Initialize stats storage using CLEANED names as keys
        stats = defaultdict(lambda: {
            "kills": 0,
            "deaths": 0,
            "first_bloods": 0,
            "first_deaths": 0,
            "plants": 0,
            "defuses": 0,
            "rounds_played": 0,
            "matches_played": 0,
            "rounds_won": 0,
            "rounds_lost": 0,
            "attack_rounds": 0,
            "defense_rounds": 0,
            "attack_wins": 0,
            "defense_wins": 0,
            "agents": Counter(),
            "maps": Counter(),
            # Track KAST per round
            "rounds_with_kill": 0,
            "rounds_with_assist": 0,
            "rounds_survived": 0,
            "rounds_traded": 0,
            "rounds_contributed": 0,  # Counts rounds where player contributed (K/A/S/T)
            "trade_kills": 0,
            "display_name": "",  # Store original name for display
        })
        
        # Process each match
        for match in filtered_matches:
            # Get allowed players in this match (raw names), map to cleaned
            match_players_raw = [
                p for p in match['players_agents'].keys()
                if self.is_allowed_player(p)
            ]
            
            if not match_players_raw:
                continue
            
            # Track agents and maps using cleaned names
            for player_raw in match_players_raw:
                player_clean = clean_player_name(player_raw)
                stats[player_clean]["matches_played"] += 1
                stats[player_clean]["agents"][match['players_agents'][player_raw]] += 1
                stats[player_clean]["maps"][match['map_name']] += 1
                # Store display name (first occurrence)
                if not stats[player_clean]["display_name"]:
                    stats[player_clean]["display_name"] = player_raw
            
            # Create set of cleaned player names for quick lookup
            match_players_clean = {clean_player_name(p) for p in match_players_raw}
            
            # Process each round
            for round_idx, round_data in enumerate(match['rounds']):
                round_won = round_data.get('outcome') == 'win'
                side = round_data.get('side', 'Unknown')
                
                # Track players alive at end
                players_alive = match_players_clean.copy()
                
                # Track kills and deaths in this round
                round_killers = set()
                round_victims = set()
                
                # Track assists (damage dealt)
                damage_events = defaultdict(set)
                heal_events = defaultdict(set)
                
                # Process events to track kills/deaths/damage/heal
                kill_events = []  # Store kills for trade calculation
                
                for event in round_data.get('events', []):
                    actor = event.get('actor')
                    target = event.get('target')
                    event_type = event.get('event_type')
                    
                    if not actor or not target:
                        continue
                    
                    actor_clean = clean_player_name(actor)
                    target_clean = clean_player_name(target)
                    
                    if event_type == 'kill':
                        if actor_clean in match_players_clean:
                            stats[actor_clean]['kills'] += 1
                            round_killers.add(actor_clean)
                        
                        if target_clean in match_players_clean:
                            stats[target_clean]['deaths'] += 1
                            round_victims.add(target_clean)
                            players_alive.discard(target_clean)
                        
                        # Store for trade calculation
                        kill_events.append({
                            'actor_clean': actor_clean,
                            'target_clean': target_clean,
                            'actor_raw': actor,
                            'target_raw': target,
                            'timestamp': event.get('timestamp', 0),
                            'is_team_kill': actor_clean in match_players_clean and target_clean in match_players_clean
                        })
                    
                    elif event_type == 'plant':
                        if actor_clean in match_players_clean:
                            stats[actor_clean]['plants'] += 1
                    
                    elif event_type == 'defuse':
                        if actor_clean in match_players_clean:
                            stats[actor_clean]['defuses'] += 1
                    
                    elif event_type == 'damage':
                        if actor_clean in match_players_clean and target_clean in match_players_clean:
                            damage_events[actor_clean].add(target_clean)
                    
                    elif event_type == 'heal':
                        if actor_clean in match_players_clean and target_clean in match_players_clean:
                            heal_events[actor_clean].add(target_clean)
                
                # Calculate trades (kills within trade window after teammate death)
                trade_beneficiaries = set()  # Players who got trade kills
                traded_players = set()       # Players who were traded (died and were avenged)

                # First pass: track who killed our teammates (only enemy kills, not team kills)
                death_details = {}  # teammate_clean -> {'killer': enemy_clean, 'time': timestamp}
                
                for kill in kill_events:
                    if not kill['is_team_kill'] and kill['target_clean'] in match_players_clean:
                        # Our teammate was killed by enemy
                        death_details[kill['target_clean']] = {
                            'killer': kill['actor_clean'],  # Enemy who killed them
                            'time': kill['timestamp']
                        }

                # Second pass: check for trades
                for kill in kill_events:
                    killer_clean = kill['actor_clean']
                    victim_clean = kill['target_clean']
                    kill_time = kill['timestamp']
                    
                    # Only consider our players killing enemies (not team kills)
                    if killer_clean in match_players_clean and not kill['is_team_kill']:
                        # Check if we killed someone who recently killed a teammate
                        for teammate, details in death_details.items():
                            enemy_killer = details['killer']
                            death_time = details['time']
                            
                            # Check: 1. Same enemy? 2. Within time window? 3. Different players?
                            if (enemy_killer == victim_clean and 
                                teammate != killer_clean and 
                                0 < (kill_time - death_time) <= Config.TRADE_WINDOW_SECONDS):
                                # This is a VALID trade!
                                stats[killer_clean]['trade_kills'] += 1
                                traded_players.add(teammate)
                                trade_beneficiaries.add(killer_clean)
                                break  # Found a trade for this kill
                
                # Calculate assists (damage/heal that contributed to kills)
                assist_players = set()
                for kill in kill_events:
                    killer_clean = kill['actor_clean']
                    victim_clean = kill['target_clean']
                    
                    # Skip team kills for assist calculation
                    if kill['is_team_kill']:
                        continue
                    
                    # Find players who damaged this victim (enemy) before they died
                    if killer_clean in match_players_clean:
                        # If one of our players got the kill, check who damaged the victim
                        for damager, damaged_players in damage_events.items():
                            if victim_clean in damaged_players and damager != killer_clean:
                                assist_players.add(damager)
                        # Find players who healed the killer (our player)
                        for healer, healed_players in heal_events.items():
                            if killer_clean in healed_players:
                                assist_players.add(healer)
                
                # Update round-based stats for all players
                for player_clean in match_players_clean:
                    stats[player_clean]["rounds_played"] += 1
                    
                    if round_won:
                        stats[player_clean]["rounds_won"] += 1
                    else:
                        stats[player_clean]["rounds_lost"] += 1
                    
                    # Check KAST components for this specific round
                    contributed_this_round = False
                    
                    # Kill
                    if player_clean in round_killers:
                        stats[player_clean]["rounds_with_kill"] += 1
                        contributed_this_round = True
                    
                    # Assist (damage or heal that led to kill)
                    if player_clean in assist_players:
                        stats[player_clean]["rounds_with_assist"] += 1
                        contributed_this_round = True
                    
                    # Survive (alive at end of round)
                    if player_clean in players_alive:
                        stats[player_clean]["rounds_survived"] += 1
                        contributed_this_round = True
                    
                    # Trade (was killed and avenged within trade window)
                    if player_clean in traded_players:
                        stats[player_clean]["rounds_traded"] += 1
                        contributed_this_round = True
                    
                    # Track if player contributed this round
                    if contributed_this_round:
                        stats[player_clean]["rounds_contributed"] += 1
                    
                    # Side stats
                    if side == 'Attack':
                        stats[player_clean]['attack_rounds'] += 1
                        if round_won:
                            stats[player_clean]['attack_wins'] += 1
                    elif side == 'Defense':
                        stats[player_clean]['defense_rounds'] += 1
                        if round_won:
                            stats[player_clean]['defense_wins'] += 1
                
                # First blood/death (using cleaned names)
                fb_player_raw = round_data.get('first_blood_player')
                fd_player_raw = round_data.get('first_death_player')
                
                if fb_player_raw:
                    fb_player_clean = clean_player_name(fb_player_raw)
                    if fb_player_clean in match_players_clean:
                        stats[fb_player_clean]['first_bloods'] += 1
                
                if fd_player_raw:
                    fd_player_clean = clean_player_name(fd_player_raw)
                    if fd_player_clean in match_players_clean:
                        stats[fd_player_clean]['first_deaths'] += 1
        
        # Calculate derived statistics
        player_stats = []
        
        for player_clean, data in stats.items():
            if data['rounds_played'] == 0:
                continue
            
            rounds = data['rounds_played']
            
            # Basic ratios
            kd = safe_division(data['kills'], data['deaths'], data['kills'])
            kpr = data['kills'] / rounds
            dpr = data['deaths'] / rounds
            fbpr = data['first_bloods'] / rounds
            fdpr = data['first_deaths'] / rounds
            
            # CORRECT KAST calculation
            # KAST% = (Rounds with Kill, Assist, Survive, or Trade) / Total Rounds Played
            kast = safe_percentage(data['rounds_contributed'], rounds)
            
            # Win rates
            overall_winrate = safe_percentage(data['rounds_won'], rounds)
            attack_winrate = safe_percentage(data['attack_wins'], data['attack_rounds'])
            defense_winrate = safe_percentage(data['defense_wins'], data['defense_rounds'])
            
            # Other rates
            survival_rate = safe_percentage(data['rounds_survived'], rounds)
            trade_rate = safe_percentage(data['trade_kills'], data['deaths']) if data['deaths'] > 0 else 0.0
            
            # Rating calculation using config weights
            impact_rating = (
                kpr * Config.RATING_WEIGHTS['kpr'] +
                (fbpr * Config.FB_IMPACT_MULTIPLIER) * Config.RATING_WEIGHTS['fb_impact'] +
                (kast / 100) * Config.RATING_WEIGHTS['kast'] +
                (overall_winrate / 100) * Config.RATING_WEIGHTS['win_contribution'] +
                (trade_rate / 100) * Config.RATING_WEIGHTS['teamplay'] +
                -dpr * Config.RATING_WEIGHTS['death_penalty']
            )
            
            player_stats.append({
                'Player': data['display_name'] or player_clean,  # Use original name if available
                'Matches': data['matches_played'],
                'Rounds': rounds,
                'Kills': data['kills'],
                'Deaths': data['deaths'],
                'Assists': data['rounds_with_assist'],  # Fixed: Now shows actual assists
                'Trade Kills': data['trade_kills'],       # New: Separate column for trade kills
                'K/D': round(kd, Config.STAT_PRECISION['kd']),
                'KPR': round(kpr, Config.STAT_PRECISION['kpr']),
                'DPR': round(dpr, Config.STAT_PRECISION['dpr']),
                'FBPR': round(fbpr, Config.STAT_PRECISION['fbpr']),
                'FDPR': round(fdpr, Config.STAT_PRECISION['fdpr']),
                'KAST%': round(kast, Config.STAT_PRECISION['percentage']),
                'Survival%': round(survival_rate, Config.STAT_PRECISION['percentage']),
                'Win%': round(overall_winrate, Config.STAT_PRECISION['percentage']),
                'Attack%': round(attack_winrate, Config.STAT_PRECISION['percentage']),
                'Defense%': round(defense_winrate, Config.STAT_PRECISION['percentage']),
                'Trade%': round(trade_rate, Config.STAT_PRECISION['percentage']),
                'Plants': data['plants'],
                'Defuses': data['defuses'],
                'First Bloods': data['first_bloods'],
                'First Deaths': data['first_deaths'],
                'Rating': round(impact_rating, 3),  # Raw rating before normalization
                'Main Agent': data['agents'].most_common(1)[0][0] if data['agents'] else 'N/A',
            })
        
        if not player_stats:
            return pd.DataFrame()
        
        # Create DataFrame and normalize ratings
        df = pd.DataFrame(player_stats)
        
        if 'Rating' in df.columns and len(df) > 0 and df['Rating'].mean() != 0:
            mean_rating = df['Rating'].mean()
            df['Rating'] = (df['Rating'] / mean_rating).round(
                Config.STAT_PRECISION['rating']
            )
        
        return df
    
    def calculate_team_stats(self, filtered_matches: List[Dict]) -> Dict:
        """Calculate team-level statistics.
        
        Args:
            filtered_matches: List of matches to analyze
            
        Returns:
            Dictionary with team statistics
        """
        team_stats = {
            'total_matches': len(filtered_matches),
            'total_rounds': 0,
            'wins': 0,
            'losses': 0,
            'rounds_won': 0,
            'rounds_lost': 0,
            'attack_rounds': 0,
            'defense_rounds': 0,
            'attack_wins': 0,
            'defense_wins': 0,
            'first_bloods': 0,
            'first_deaths': 0,
            'plants': 0,
            'defuses': 0,
            'pistol_rounds': 0,
            'pistol_wins': 0,
            'eco_rounds': 0,
            'eco_wins': 0,
            'anti_eco_rounds': 0,
            'anti_eco_wins': 0,
            'full_buy_rounds': 0,
            'full_buy_wins': 0,
            'post_plant_rounds': 0,
            'post_plant_wins': 0,
            'retake_attempts': 0,
            'retake_wins': 0,
            'site_stats': defaultdict(lambda: {'attempts': 0, 'wins': 0}),
            'map_stats': defaultdict(lambda: {'played': 0, 'wins': 0}),
            'man_advantage': defaultdict(lambda: {'occurrences': 0, 'wins': 0}),
        }
        
        for match in filtered_matches:
            map_name = match['map_name']
            team_stats['map_stats'][map_name]['played'] += 1
            
            # Get allowed players (cleaned names)
            match_players = {
                clean_player_name(p) for p in match['players_agents'].keys()
                if self.is_allowed_player(p)
            }
            
            if not match_players:
                continue
            
            # Check match outcome
            score = match.get('final_score', '').split(' - ')
            if len(score) == 2:
                try:
                    team_score = int(score[0].strip())
                    opp_score = int(score[1].strip())
                    if team_score > opp_score:
                        team_stats['wins'] += 1
                        team_stats['map_stats'][map_name]['wins'] += 1
                    else:
                        team_stats['losses'] += 1
                except (ValueError, IndexError):
                    pass
            
            team_stats['total_rounds'] += len(match['rounds'])
            
            # Process rounds
            for round_data in match['rounds']:
                round_won = round_data.get('outcome') == 'win'
                side = round_data.get('side', 'Unknown')
                round_num = round_data.get('round_number', 0)
                
                if round_won:
                    team_stats['rounds_won'] += 1
                else:
                    team_stats['rounds_lost'] += 1
                
                # Side tracking
                if side == 'Attack':
                    team_stats['attack_rounds'] += 1
                    if round_won:
                        team_stats['attack_wins'] += 1
                elif side == 'Defense':
                    team_stats['defense_rounds'] += 1
                    if round_won:
                        team_stats['defense_wins'] += 1
                
                # First blood tracking (FIXED: Separate ifs, validation for both)
                fb_player = round_data.get('first_blood_player')
                fd_player = round_data.get('first_death_player')
                
                if fb_player and self.is_allowed_player(fb_player):
                    team_stats['first_bloods'] += 1
                
                if fd_player and self.is_allowed_player(fd_player):
                    team_stats['first_deaths'] += 1
                
                # Plant/defuse tracking
                for event in round_data.get('events', []):
                    actor = event.get('actor')
                    event_type = event.get('event_type')
                    
                    if actor:
                        actor_clean = clean_player_name(actor)
                        if actor_clean in match_players:
                            if event_type == 'plant':
                                team_stats['plants'] += 1
                            elif event_type == 'defuse':
                                team_stats['defuses'] += 1
                
                # Economy analysis (FIXED: Type checking)
                try:
                    team_econ_val = round_data.get('team_economy', '0')
                    opp_econ_val = round_data.get('opponent_economy', '0')
                    
                    # Handle both string and int types
                    if isinstance(team_econ_val, str):
                        team_econ = int(team_econ_val.replace(',', ''))
                    else:
                        team_econ = int(team_econ_val)
                        
                    if isinstance(opp_econ_val, str):
                        opp_econ = int(opp_econ_val.replace(',', ''))
                    else:
                        opp_econ = int(opp_econ_val)
                    
                    # Pistol rounds
                    if round_num in [1, 2, 13, 14]:
                        team_stats['pistol_rounds'] += 1
                        if round_won:
                            team_stats['pistol_wins'] += 1
                    
                    # Eco rounds
                    if team_econ < Config.ECO_ROUND_THRESHOLD:
                        team_stats['eco_rounds'] += 1
                        if round_won:
                            team_stats['eco_wins'] += 1
                    
                    # Anti-eco
                    if team_econ >= Config.FULL_BUY_THRESHOLD and opp_econ < Config.ECO_ROUND_THRESHOLD:
                        team_stats['anti_eco_rounds'] += 1
                        if round_won:
                            team_stats['anti_eco_wins'] += 1
                    
                    # Full buy
                    if team_econ >= Config.FULL_BUY_THRESHOLD:
                        team_stats['full_buy_rounds'] += 1
                        if round_won:
                            team_stats['full_buy_wins'] += 1
                
                except (ValueError, AttributeError, TypeError):
                    pass
                
                # Post-plant tracking
                has_plant = round_data.get('plant', False)
                if has_plant:
                    team_stats['post_plant_rounds'] += 1
                    if round_won and side == 'Attack':
                        team_stats['post_plant_wins'] += 1
                    
                    # Retake tracking
                    if side == 'Defense':
                        team_stats['retake_attempts'] += 1
                        if round_won:
                            team_stats['retake_wins'] += 1
                
                # Site tracking
                site = round_data.get('site')
                if site and site not in ['unclear', 'null', '', None]:
                    team_stats['site_stats'][site]['attempts'] += 1
                    if round_won:
                        team_stats['site_stats'][site]['wins'] += 1
        
        return team_stats
    
    def calculate_round_timing_stats(self, filtered_matches: List[Dict]) -> pd.DataFrame:
        """Calculate timing-based statistics for all rounds.
        
        Args:
            filtered_matches: List of matches to analyze
            
        Returns:
            DataFrame with timing data including outcome
        """
        timing_data = []
        
        for match in filtered_matches:
            for round_data in match['rounds']:
                # Get max timestamp as round duration
                timestamps = [
                    e.get('timestamp', 0)
                    for e in round_data.get('events', [])
                ]
                
                if not timestamps:
                    continue
                
                duration = max(timestamps)
                
                # Only include rounds up to max duration
                if duration <= Config.ROUND_MAX_DURATION:
                    timing_data.append({
                        'match_id': match.get('id', match.get('filename', 'unknown')),
                        'round': round_data.get('round_number', 0),
                        'duration': duration,
                        'map': match['map_name'],
                        'side': round_data.get('side', 'Unknown'),
                        'first_blood': round_data.get('first_blood', False),
                        'plant': round_data.get('plant', False),
                        'outcome': round_data.get('outcome', 'loss')  # Add outcome
                    })
        
        return pd.DataFrame(timing_data) if timing_data else pd.DataFrame()


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Valorant Team Analytics Dashboard",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Title
    st.title("🎯 Valorant Team Analytics Dashboard")
    
    # Initialize analyzer
    analyzer = ValorantAnalyzer(data_folder="data")
    
    if not analyzer.matches:
        st.warning("⚠️ No match data found. Please add JSON files to the data folder.")
        return
    
    # ========================================================================
    # SIDEBAR FILTERS
    # ========================================================================
    
    st.sidebar.header("Filters")
    
    # Date range filter
    all_dates = [m['parsed_date'] for m in analyzer.matches]
    min_date = min(all_dates).date()
    max_date = max(all_dates).date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date = datetime.combine(date_range[0], datetime.min.time())
        end_date = datetime.combine(date_range[1], datetime.max.time())
    else:
        start_date, end_date = None, None
    
    # Quick time filters
    time_filter = st.sidebar.selectbox(
        "Quick Filters",
        ["All Time", "Last 7 Days", "Last 14 Days", "Last 10 Matches", "Last 20 Matches"]
    )
    
    last_n_matches = None
    if time_filter == "Last 7 Days":
        start_date = datetime.now() - timedelta(days=7)
    elif time_filter == "Last 14 Days":
        start_date = datetime.now() - timedelta(days=14)
    elif time_filter == "Last 10 Matches":
        last_n_matches = 10
    elif time_filter == "Last 20 Matches":
        last_n_matches = 20
    
    # Map filter
    all_maps = analyzer.get_all_maps()
    selected_maps = st.sidebar.multiselect(
        "Maps",
        options=all_maps,
        default=all_maps
    )
    
    # Player filter (now returns cleaned names)
    all_players = analyzer.get_all_players()
    selected_players = st.sidebar.multiselect(
        "Players",
        options=all_players,
        default=all_players
    )
    
    # Apply filters
    filtered_matches = analyzer.filter_matches(
        selected_maps=selected_maps,
        selected_players=selected_players,
        start_date=start_date,
        end_date=end_date,
        last_n_matches=last_n_matches
    )
    
    if not filtered_matches:
        st.warning("⚠️ No matches found with the selected filters.")
        return
    
    # ========================================================================
    # MAIN TABS
    # ========================================================================
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "👤 Player Stats",
        "🎯 Round Analysis",
        "💰 Economy",
        "📍 Site Analysis",
        "📈 Trends"
    ])
    
    # ------------------------------------------------------------------------
    # TAB 1: OVERVIEW
    # ------------------------------------------------------------------------
    
    with tab1:
        st.header("Team Overview")
        
        team_stats = analyzer.calculate_team_stats(filtered_matches)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Matches Played", team_stats['total_matches'])
            match_winrate = safe_percentage(
                team_stats['wins'],
                team_stats['wins'] + team_stats['losses']
            )
            st.metric("Match Win Rate", f"{match_winrate:.1f}%")
        
        with col2:
            overall_winrate = safe_percentage(
                team_stats['rounds_won'],
                team_stats['total_rounds']
            )
            st.metric("Round Win %", f"{overall_winrate:.1f}%")
            
            attack_winrate = safe_percentage(
                team_stats['attack_wins'],
                team_stats['attack_rounds']
            )
            st.metric("Attack Win %", f"{attack_winrate:.1f}%")
        
        with col3:
            defense_winrate = safe_percentage(
                team_stats['defense_wins'],
                team_stats['defense_rounds']
            )
            st.metric("Defense Win %", f"{defense_winrate:.1f}%")
            
            eco_winrate = safe_percentage(
                team_stats['eco_wins'],
                team_stats['eco_rounds']
            )
            st.metric("Eco Win %", f"{eco_winrate:.1f}%")
        
        with col4:
            anti_eco_winrate = safe_percentage(
                team_stats['anti_eco_wins'],
                team_stats['anti_eco_rounds']
            )
            st.metric("Anti-Eco Win %", f"{anti_eco_winrate:.1f}%")
            
            post_plant_winrate = safe_percentage(
                team_stats['post_plant_wins'],
                team_stats['post_plant_rounds']
            )
            st.metric("Post-Plant Win %", f"{post_plant_winrate:.1f}%")
        
        # Recent matches
        st.subheader("Recent Matches")
        matches_table = []
        for match in filtered_matches[-10:]:
            match_players = [
                p for p in match['players_agents'].keys()
                if analyzer.is_allowed_player(p)
            ]
            player_str = ', '.join(match_players[:3])
            if len(match_players) > 3:
                player_str += '...'
            
            matches_table.append({
                'Date': match['date'],
                'Map': match['map_name'],
                'Score': match.get('final_score', 'N/A'),
                'Players': player_str
            })
        
        if matches_table:
            st.dataframe(pd.DataFrame(matches_table), use_container_width=True)
        
        # Map performance
        st.subheader("Map Performance")
        map_data = []
        for map_name, stats in team_stats['map_stats'].items():
            if stats['played'] > 0:
                winrate = safe_percentage(stats['wins'], stats['played'])
                map_data.append({
                    'Map': map_name,
                    'Played': stats['played'],
                    'Wins': stats['wins'],
                    'Win %': f"{winrate:.1f}%",
                    'Win_Rate': winrate
                })
        
        if map_data:
            df_map = pd.DataFrame(map_data).sort_values('Win_Rate', ascending=False)
            st.dataframe(
                df_map[['Map', 'Played', 'Wins', 'Win %']],
                use_container_width=True
            )
    
    # ------------------------------------------------------------------------
    # TAB 2: PLAYER STATS
    # ------------------------------------------------------------------------
    
    with tab2:
        st.header("Player Statistics")
        
        player_stats_df = analyzer.calculate_player_stats(filtered_matches)
        
        if not player_stats_df.empty:
            # Sort options
            sort_by = st.selectbox(
                "Sort by",
                options=['Rating', 'K/D', 'KPR', 'KAST%', 'Win%', 'First Bloods', 'Trade Kills'],
                index=0
            )
            
            player_stats_df = player_stats_df.sort_values(sort_by, ascending=False)
            
            # Display stats
            st.dataframe(
                player_stats_df.style.format({
                    'K/D': '{:.2f}',
                    'KPR': '{:.3f}',
                    'DPR': '{:.3f}',
                    'FBPR': '{:.3f}',
                    'FDPR': '{:.3f}',
                    'KAST%': '{:.1f}',
                    'Survival%': '{:.1f}',
                    'Win%': '{:.1f}',
                    'Attack%': '{:.1f}',
                    'Defense%': '{:.1f}',
                    'Trade%': '{:.1f}',
                    'Rating': '{:.2f}'
                }),
                use_container_width=True,
                height=400
            )
            
            # Visualizations
            st.subheader("Player Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    player_stats_df.head(10),
                    x='Player',
                    y='Rating',
                    title='Top 10 Players by Rating',
                    color='Rating',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    player_stats_df,
                    x='KAST%',
                    y='Win%',
                    size='Rounds',
                    color='Rating',
                    hover_name='Player',
                    title='KAST% vs Win Rate',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Role performance
            st.subheader("Agent Performance")
            if 'Main Agent' in player_stats_df.columns:
                role_stats = player_stats_df.groupby('Main Agent').agg({
                    'Player': 'count',
                    'Rating': 'mean',
                    'K/D': 'mean',
                    'KAST%': 'mean',
                    'Win%': 'mean'
                }).round(2).reset_index()
                
                role_stats.columns = ['Agent', 'Players', 'Avg Rating', 'Avg K/D', 'Avg KAST%', 'Avg Win%']
                role_stats = role_stats.sort_values('Avg Rating', ascending=False)
                
                st.dataframe(role_stats, use_container_width=True)
        else:
            st.info("No player data available")
    
    # ------------------------------------------------------------------------
    # TAB 3: ROUND ANALYSIS
    # ------------------------------------------------------------------------
    
    with tab3:
        st.header("Round Analysis")
        
        team_stats = analyzer.calculate_team_stats(filtered_matches)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("First Blood Impact")
            st.metric("First Bloods", team_stats['first_bloods'])
            st.metric("First Deaths", team_stats['first_deaths'])
            
            fb_diff = team_stats['first_bloods'] - team_stats['first_deaths']
            st.metric("FB Differential", f"{fb_diff:+d}")
        
        with col2:
            st.subheader("Round Timing Analysis")
            timing_df = analyzer.calculate_round_timing_stats(filtered_matches)
            
            if not timing_df.empty:
                avg_duration = timing_df['duration'].mean()
                st.metric("Avg Round Duration", f"{avg_duration:.1f}s")
                
                # Duration distribution
                fig = px.histogram(
                    timing_df,
                    x='duration',
                    nbins=30,
                    title='Round Duration Distribution (All Rounds)',
                    labels={'duration': 'Duration (seconds)', 'count': 'Frequency'},
                    color='outcome',
                    color_discrete_map={'win': '#00FF00', 'loss': '#FF4655'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No timing data available")
        
        # Round duration to win rate analysis - FIXED VERSION
        st.subheader("Round Duration vs Win Rate")
        if not timing_df.empty:
            # Create bins for duration (10-second intervals)
            timing_df['duration_bin'] = pd.cut(
                timing_df['duration'],
                bins=range(0, Config.ROUND_MAX_DURATION + 10, 10),
                labels=[f"{i}-{i+10}s" for i in range(0, Config.ROUND_MAX_DURATION, 10)]
            )
            
            # Calculate win rate per bin
            winrate_by_duration = timing_df.groupby('duration_bin').agg(
                total_rounds=('duration', 'count'),
                wins=('outcome', lambda x: (x == 'win').sum())
            ).reset_index()
            
            winrate_by_duration['win_rate'] = (
                winrate_by_duration['wins']
                    .div(winrate_by_duration['total_rounds'])
                    .mul(100)
                    .fillna(0.0)
            )
            
            # Filter out bins with too few rounds
            winrate_by_duration = winrate_by_duration[winrate_by_duration['total_rounds'] >= 3]
            
            if not winrate_by_duration.empty:
                # Bar chart version
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_bar = px.bar(
                        winrate_by_duration,
                        x='duration_bin',
                        y='win_rate',
                        title='Win Rate by Round Duration (Bar)',
                        labels={'duration_bin': 'Duration Range', 'win_rate': 'Win Rate %'},
                        text=[f"{w:.1f}%" for w in winrate_by_duration['win_rate']],
                        color='win_rate',
                        color_continuous_scale='RdYlGn'
                    )
                    fig_bar.update_traces(
                        texttemplate='%{text}',
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Win Rate: %{y:.1f}%<br>Rounds: %{customdata[0]}<extra></extra>',
                        customdata=winrate_by_duration[['total_rounds']].values
                    )
                    fig_bar.update_layout(
                        yaxis_title='Win Rate %',
                        yaxis_range=[0, 100]
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    # Line chart version (smoothed)
                    # Sort by bin midpoint for proper line connection
                    winrate_by_duration = winrate_by_duration.sort_values('duration_bin')
                    
                    # Extract midpoint of each bin for x-axis
                    bin_midpoints = []
                    for bin_label in winrate_by_duration['duration_bin']:
                        if isinstance(bin_label, str):
                            start = int(bin_label.split('-')[0])
                            end = int(bin_label.split('-')[1].replace('s', ''))
                            bin_midpoints.append((start + end) / 2)
                        else:
                            bin_midpoints.append(0)
                    
                    winrate_by_duration['bin_midpoint'] = bin_midpoints
                    
                    # Create smoothed line using rolling average
                    winrate_sorted = winrate_by_duration.sort_values('bin_midpoint')
                    winrate_sorted['smoothed_win_rate'] = winrate_sorted['win_rate'].rolling(
                        window=2, center=True, min_periods=1
                    ).mean()
                    
                    fig_line = go.Figure()
                    
                    # Add smoothed line
                    fig_line.add_trace(go.Scatter(
                        x=winrate_sorted['bin_midpoint'],
                        y=winrate_sorted['smoothed_win_rate'],
                        mode='lines+markers',
                        name='Smoothed Win Rate',
                        line=dict(color='#FF4655', width=3),
                        marker=dict(size=8),
                        hovertemplate='<b>%{x:.0f}s</b><br>Win Rate: %{y:.1f}%<br>Rounds: %{customdata}<extra></extra>',
                        customdata=winrate_sorted['total_rounds']
                    ))
                    
                    # Add actual data points
                    fig_line.add_trace(go.Scatter(
                        x=winrate_sorted['bin_midpoint'],
                        y=winrate_sorted['win_rate'],
                        mode='markers',
                        name='Actual Win Rate',
                        marker=dict(
                            size=10,
                            color=winrate_sorted['win_rate'],
                            colorscale='RdYlGn',
                            showscale=False,
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate='<b>%{x:.0f}s</b><br>Win Rate: %{y:.1f}%<br>Rounds: %{customdata}<extra></extra>',
                        customdata=winrate_sorted['total_rounds']
                    ))
                    
                    fig_line.update_layout(
                        title='Win Rate by Round Duration (Smoothed Line)',
                        xaxis_title='Round Duration (seconds)',
                        yaxis_title='Win Rate %',
                        yaxis_range=[0, 100],
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_line, use_container_width=True)
                
                # Stats summary
                st.caption(f"Analysis based on {len(timing_df)} rounds. Showing only duration ranges with 3+ rounds.")
            else:
                st.info("Not enough data for duration analysis (need at least 3 rounds per bin)")
    
    # ------------------------------------------------------------------------
    # TAB 4: ECONOMY
    # ------------------------------------------------------------------------
    
    with tab4:
        st.header("Economic Analysis")
        
        team_stats = analyzer.calculate_team_stats(filtered_matches)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pistol_wr = safe_percentage(
                team_stats['pistol_wins'],
                team_stats['pistol_rounds']
            )
            st.metric(
                "Pistol Rounds",
                f"{team_stats['pistol_wins']}/{team_stats['pistol_rounds']}",
                f"{pistol_wr:.1f}%"
            )
        
        with col2:
            eco_wr = safe_percentage(team_stats['eco_wins'], team_stats['eco_rounds'])
            st.metric(
                "Eco Rounds",
                f"{team_stats['eco_wins']}/{team_stats['eco_rounds']}",
                f"{eco_wr:.1f}%"
            )
        
        with col3:
            full_buy_wr = safe_percentage(
                team_stats['full_buy_wins'],
                team_stats['full_buy_rounds']
            )
            st.metric(
                "Full Buy Rounds",
                f"{team_stats['full_buy_wins']}/{team_stats['full_buy_rounds']}",
                f"{full_buy_wr:.1f}%"
            )
        
        # Economic chart
        st.subheader("Economic Performance")
        
        eco_data = {
            'Round Type': ['Pistol', 'Eco', 'Anti-Eco', 'Full Buy'],
            'Rounds': [
                team_stats['pistol_rounds'],
                team_stats['eco_rounds'],
                team_stats['anti_eco_rounds'],
                team_stats['full_buy_rounds']
            ],
            'Wins': [
                team_stats['pistol_wins'],
                team_stats['eco_wins'],
                team_stats['anti_eco_wins'],
                team_stats['full_buy_wins']
            ]
        }
        
        win_rates = [
            safe_percentage(eco_data['Wins'][i], eco_data['Rounds'][i])
            for i in range(len(eco_data['Round Type']))
        ]
        eco_data['Win Rate'] = win_rates
        
        fig = px.bar(
            pd.DataFrame(eco_data),
            x='Round Type',
            y='Win Rate',
            title='Win Rate by Round Type',
            color='Win Rate',
            color_continuous_scale='RdYlGn',
            text=[f"{w:.1f}%" for w in win_rates]
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # ------------------------------------------------------------------------
    # TAB 5: SITE ANALYSIS
    # ------------------------------------------------------------------------
    
    with tab5:
        st.header("Site Analysis")
        
        team_stats = analyzer.calculate_team_stats(filtered_matches)
        
        if team_stats['site_stats']:
            site_data = []
            for site, stats in team_stats['site_stats'].items():
                if stats['attempts'] > 0:
                    winrate = safe_percentage(stats['wins'], stats['attempts'])
                    site_data.append({
                        'Site': site,
                        'Attempts': stats['attempts'],
                        'Wins': stats['wins'],
                        'Win %': f"{winrate:.1f}%",
                        'Win Rate': winrate
                    })
            
            if site_data:
                site_df = pd.DataFrame(site_data).sort_values('Attempts', ascending=False)
                
                st.dataframe(
                    site_df[['Site', 'Attempts', 'Wins', 'Win %']],
                    use_container_width=True
                )
                
                # Visualization
                fig = px.bar(
                    site_df,
                    x='Site',
                    y='Win Rate',
                    title='Site Win Rates',
                    color='Win Rate',
                    color_continuous_scale='RdYlGn',
                    text=[f"{w:.1f}%" for w in site_df['Win Rate']]
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No site data available")
        
        # Retake analysis
        st.subheader("Retake Performance")
        
        if team_stats['retake_attempts'] > 0:
            retake_wr = safe_percentage(
                team_stats['retake_wins'],
                team_stats['retake_attempts']
            )
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Retake Attempts", team_stats['retake_attempts'])
            with col2:
                st.metric("Retake Win %", f"{retake_wr:.1f}%")
        else:
            st.info("No retake data available")
    
    # ------------------------------------------------------------------------
    # TAB 6: TRENDS
    # ------------------------------------------------------------------------
    
    with tab6:
        st.header("Performance Trends")
        
        # Calculate stats over time
        time_data = []
        
        for match in filtered_matches:
            match_stats = analyzer.calculate_player_stats([match])
            
            if not match_stats.empty:
                avg_rating = match_stats['Rating'].mean()
                avg_kd = match_stats['K/D'].mean()
                avg_winrate = match_stats['Win%'].mean()
                
                time_data.append({
                    'Date': match['parsed_date'],
                    'Map': match['map_name'],
                    'Score': match.get('final_score', 'N/A'),
                    'Avg Rating': avg_rating,
                    'Avg K/D': avg_kd,
                    'Avg Win Rate': avg_winrate
                })
        
        if time_data:
            trends_df = pd.DataFrame(time_data).sort_values('Date')
            
            # Rating over time
            fig = px.line(
                trends_df,
                x='Date',
                y='Avg Rating',
                title='Average Team Rating Over Time',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Map-based trends
            if len(trends_df['Map'].unique()) > 1:
                fig = px.line(
                    trends_df,
                    x='Date',
                    y='Avg Rating',
                    color='Map',
                    title='Rating Trends by Map',
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Match-by-match table
            st.subheader("Match-by-Match Performance")
            display_df = trends_df[[
                'Date', 'Map', 'Score', 'Avg Rating', 'Avg K/D', 'Avg Win Rate'
            ]].copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No trend data available")
    
    # Footer
    st.sidebar.markdown("---")
    total_wins = team_stats['wins']
    total_losses = team_stats['losses']
    st.sidebar.info(
        f"Showing {len(filtered_matches)} matches "
        f"({total_wins}W - {total_losses}L)"
    )


if __name__ == "__main__":
    main()