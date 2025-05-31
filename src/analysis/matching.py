"""
HLA Matching Analysis
-------------------
Tools for analyzing HLA matching between donors and recipients.
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Set
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional imports for visualizations and reporting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib/Seaborn not installed; visualization features will be limited")
    PLOTTING_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    PDF_AVAILABLE = True
except ImportError:
    logger.warning("ReportLab not installed; PDF report generation will not be available")
    PDF_AVAILABLE = False

class MatchingAnalyzer:
    """Analyzes HLA matching between donors and recipients"""
    
    # Standard loci to consider for matching analysis
    STANDARD_LOCI = ["A", "B", "C", "DRB1", "DQB1", "DPB1"]
    
    # Standard importance weights for different loci
    LOCUS_WEIGHTS = {
        "A": 1.0,
        "B": 1.0,
        "C": 0.8,
        "DRB1": 1.0,
        "DQB1": 0.8,
        "DPB1": 0.6
    }
    
    def __init__(
        self, 
        encoder, 
        loci: Optional[List[str]] = None,
        locus_weights: Optional[Dict[str, float]] = None,
        similarity_threshold: float = 0.9
    ):
        """Initialize matching analyzer
        
        Args:
            encoder: HLAEncoder instance used to generate embeddings
            loci: List of HLA loci to consider (defaults to standard loci)
            locus_weights: Dict mapping loci to importance weights
            similarity_threshold: Threshold for considering alleles functionally similar
        """
        self.encoder = encoder
        self.loci = loci or self.STANDARD_LOCI
        self.locus_weights = locus_weights or self.LOCUS_WEIGHTS
        self.similarity_threshold = similarity_threshold
    
    def group_alleles_by_locus(self, alleles: List[str]) -> Dict[str, List[str]]:
        """Group alleles by locus
        
        Args:
            alleles: List of HLA allele names
            
        Returns:
            Dict mapping locus to list of alleles
        """
        grouped = {}
        
        for allele in alleles:
            try:
                # Parse locus from allele
                if '*' in allele:
                    locus = allele.split('*')[0]
                else:
                    locus = ''.join(c for c in allele if c.isalpha())
                    
                if locus not in grouped:
                    grouped[locus] = []
                    
                grouped[locus].append(allele)
            except Exception as e:
                logger.warning(f"Error parsing locus for allele {allele}: {e}")
        
        return grouped
    
    def find_best_match(self, allele: str, candidates: List[str]) -> Tuple[str, float]:
        """Find best matching allele from candidates
        
        Args:
            allele: Target allele 
            candidates: List of candidate alleles to match against
            
        Returns:
            Tuple of (best_match_allele, similarity_score)
        """
        if not candidates:
            return None, 0.0
            
        # First check for exact match
        if allele in candidates:
            return allele, 1.0
            
        # Get embedding for target allele
        try:
            target_embedding = self.encoder.get_embedding(allele)
        except Exception as e:
            logger.error(f"Error getting embedding for {allele}: {e}")
            return None, 0.0
            
        # Compare to each candidate
        best_match = None
        best_score = -1.0
        
        for candidate in candidates:
            try:
                candidate_embedding = self.encoder.get_embedding(candidate)
                similarity = self.encoder._cosine_similarity(target_embedding, candidate_embedding)
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = candidate
            except Exception as e:
                logger.warning(f"Error comparing {allele} with {candidate}: {e}")
                
        return best_match, best_score
    
    def analyze_matching(self, donor_alleles: List[str], recipient_alleles: List[str]) -> Dict:
        """Analyze matching between donor and recipient HLA
        
        Args:
            donor_alleles: List of donor HLA alleles
            recipient_alleles: List of recipient HLA alleles
            
        Returns:
            Dict with matching analysis results
        """
        # Group alleles by locus
        donor_by_locus = self.group_alleles_by_locus(donor_alleles)
        recipient_by_locus = self.group_alleles_by_locus(recipient_alleles)
        
        # Analyze matching for each locus
        exact_matches = []
        functional_matches = []
        mismatches = []
        similarity_scores = []
        
        # Common loci present in both donor and recipient
        common_loci = set(donor_by_locus.keys()) & set(recipient_by_locus.keys())
        
        # Filter to requested loci
        if self.loci:
            common_loci = common_loci & set(self.loci)
        
        # Analyze each locus
        locus_results = {}
        for locus in common_loci:
            locus_exact = 0
            locus_functional = 0
            locus_mismatches = []
            locus_similarities = []
            
            # Get weight for this locus
            locus_weight = self.locus_weights.get(locus, 1.0)
            
            # Compare each recipient allele to donor alleles
            for r_allele in recipient_by_locus[locus]:
                # Find best donor match
                best_match, similarity = self.find_best_match(r_allele, donor_by_locus[locus])
                
                # Record match type
                locus_similarities.append(similarity * locus_weight)
                
                if best_match == r_allele:
                    # Exact match
                    exact_matches.append((locus, r_allele))
                    locus_exact += 1
                elif similarity >= self.similarity_threshold:
                    # Functional match (high similarity)
                    functional_matches.append((locus, r_allele, best_match, similarity))
                    locus_functional += 1
                else:
                    # Mismatch
                    mismatches.append((r_allele, best_match, similarity))
                    locus_mismatches.append((r_allele, best_match, similarity))
            
            # Store locus-specific results
            locus_results[locus] = {
                'exact_matches': locus_exact,
                'functional_matches': locus_functional,
                'total_alleles': len(recipient_by_locus[locus]),
                'mismatches': locus_mismatches,
                'average_similarity': np.mean(locus_similarities) if locus_similarities else 0.0
            }
            
            # Add to overall scores
            similarity_scores.extend(locus_similarities)
        
        # Compute overall results
        average_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        
        # Calculate matching percentages
        total_recipient_alleles = sum(len(alleles) for locus, alleles in recipient_by_locus.items() if locus in common_loci)
        exact_match_pct = len(exact_matches) / total_recipient_alleles if total_recipient_alleles > 0 else 0
        functional_match_pct = (len(exact_matches) + len(functional_matches)) / total_recipient_alleles if total_recipient_alleles > 0 else 0
        
        # Prepare final results
        results = {
            'exact_matches': exact_matches,
            'functional_matches': functional_matches,
            'mismatches': mismatches,
            'average_similarity': average_similarity,
            'exact_match_pct': exact_match_pct,
            'functional_match_pct': functional_match_pct,
            'locus_results': locus_results,
            'common_loci': list(common_loci)
        }
        
        return results
    
    def generate_matching_report(self, results: Dict) -> pd.DataFrame:
        """Generate a detailed matching report dataframe
        
        Args:
            results: Results from analyze_matching
            
        Returns:
            DataFrame with detailed matching information
        """
        # Prepare data for report
        rows = []
        
        # Overall summary
        rows.append({
            'Locus': 'ALL',
            'Exact Matches': len(results['exact_matches']),
            'Functional Matches': len(results['functional_matches']),
            'Total Match %': f"{results['functional_match_pct']*100:.1f}%",
            'Average Similarity': f"{results['average_similarity']:.3f}"
        })
        
        # Per-locus breakdown
        for locus, locus_data in results['locus_results'].items():
            total_matches = locus_data['exact_matches'] + locus_data['functional_matches']
            match_pct = total_matches / locus_data['total_alleles'] if locus_data['total_alleles'] > 0 else 0
            
            rows.append({
                'Locus': locus,
                'Exact Matches': locus_data['exact_matches'],
                'Functional Matches': locus_data['functional_matches'],
                'Total Match %': f"{match_pct*100:.1f}%",
                'Average Similarity': f"{locus_data['average_similarity']:.3f}"
            })
        
        # Create DataFrame
        report_df = pd.DataFrame(rows)
        
        return report_df
    
    def generate_report(
        self, 
        donor_alleles: List[str], 
        recipient_alleles: List[str], 
        output_file: Optional[str] = None,
        title: str = "HLA Matching Analysis Report"
    ) -> Optional[pd.DataFrame]:
        """Generate a complete matching report
        
        Args:
            donor_alleles: List of donor HLA alleles
            recipient_alleles: List of recipient HLA alleles
            output_file: Output file path for PDF report (optional)
            title: Report title
            
        Returns:
            DataFrame with report data (if no output_file specified)
            None if output_file is specified (report saved to file)
        """
        # Run matching analysis
        results = self.analyze_matching(donor_alleles, recipient_alleles)
        
        # Generate report dataframe
        report_df = self.generate_matching_report(results)
        
        # If no output file, return the dataframe
        if output_file is None:
            return report_df
            
        # Check if PDF generation is available
        if not PDF_AVAILABLE:
            logger.error("ReportLab not installed; cannot generate PDF report")
            return report_df
            
        try:
            # Create PDF
            doc = SimpleDocTemplate(output_file, pagesize=letter)
            elements = []
            
            # Add title
            styles = getSampleStyleSheet()
            elements.append(Paragraph(title, styles['Title']))
            elements.append(Spacer(1, 12))
            
            # Add summary section
            elements.append(Paragraph("Summary", styles['Heading2']))
            elements.append(Spacer(1, 6))
            
            # Summary table
            summary_data = [
                ["Metric", "Value"],
                ["Exact Match %", f"{len(results['exact_matches']) / len(recipient_alleles) * 100:.1f}%"],
                ["Functional Match %", f"{results['functional_match_pct'] * 100:.1f}%"],
                ["Average Similarity", f"{results['average_similarity']:.3f}"],
                ["Common Loci", ", ".join(results['common_loci'])]
            ]
            
            summary_table = Table(summary_data, colWidths=[200, 200])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (1, 0), 12),
                ('BACKGROUND', (0, 1), (1, -1), colors.beige),
                ('GRID', (0, 0), (1, -1), 1, colors.black)
            ]))
            
            elements.append(summary_table)
            elements.append(Spacer(1, 12))
            
            # Per-locus breakdown
            elements.append(Paragraph("Locus Breakdown", styles['Heading2']))
            elements.append(Spacer(1, 6))
            
            # Convert dataframe to table data
            locus_data = [report_df.columns.tolist()]
            for _, row in report_df.iterrows():
                locus_data.append(row.tolist())
            
            locus_table = Table(locus_data, colWidths=[80, 80, 100, 100, 100])
            locus_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, 1), colors.lightgrey),  # Highlight ALL row
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(locus_table)
            elements.append(Spacer(1, 12))
            
            # Add mismatch details if any
            if results['mismatches']:
                elements.append(Paragraph("Mismatches Detail", styles['Heading2']))
                elements.append(Spacer(1, 6))
                
                mismatch_data = [["Recipient Allele", "Best Donor Match", "Similarity"]]
                for r_allele, d_allele, similarity in results['mismatches']:
                    mismatch_data.append([r_allele, d_allele or "None", f"{similarity:.3f}"])
                
                mismatch_table = Table(mismatch_data, colWidths=[150, 150, 100])
                mismatch_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                elements.append(mismatch_table)
            
            # Add visualization if available
            if PLOTTING_AVAILABLE:
                elements.append(Spacer(1, 12))
                elements.append(Paragraph("Visualizations", styles['Heading2']))
                elements.append(Spacer(1, 6))
                
                # Create visualization and save to a temporary file with absolute path
                temp_viz_path = os.path.abspath("temp_matching_viz.png")
                try:
                    fig = self.visualize_matching(results, output_file=temp_viz_path)
                    logger.info(f"Visualization saved to temporary file: {temp_viz_path}")
                except Exception as viz_error:
                    logger.error(f"Error generating visualization: {viz_error}")
                    # Continue without visualization
                
                # Add image to report if file exists
                try:
                    if os.path.exists(temp_viz_path):
                        from reportlab.platypus import Image
                        elements.append(Image(temp_viz_path, width=450, height=300))
                    else:
                        elements.append(Paragraph("Visualization could not be generated.", styles['Normal']))
                except Exception as img_error:
                    logger.error(f"Error adding image to report: {img_error}")
                    elements.append(Paragraph("Error including visualization in report.", styles['Normal']))
                
                # Clean up temp file - do this AFTER building the PDF
                try:
                    if os.path.exists(temp_viz_path):
                        os.remove(temp_viz_path)
                        logger.debug(f"Removed temporary visualization file: {temp_viz_path}")
                except Exception as cleanup_err:
                    logger.warning(f"Failed to remove temporary file {temp_viz_path}: {cleanup_err}")
            
            # Build the PDF
            doc.build(elements)
            logger.info(f"Matching report saved to {output_file}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return report_df
    
    def visualize_matching(
        self, 
        results: Dict, 
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """Visualize matching results
        
        Args:
            results: Results from analyze_matching
            output_file: Path to save visualization
            figsize: Figure size
            
        Returns:
            Matplotlib figure if output_file is None, otherwise None
        """
        if not PLOTTING_AVAILABLE:
            logger.error("Matplotlib not installed; cannot create visualization")
            return None
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # 1. Match percentage by locus
        loci = []
        exact_pcts = []
        functional_pcts = []
        
        for locus, locus_data in results['locus_results'].items():
            total = locus_data['total_alleles']
            if total == 0:
                continue
                
            loci.append(locus)
            exact_pcts.append(locus_data['exact_matches'] / total * 100)
            func_pct = (locus_data['exact_matches'] + locus_data['functional_matches']) / total * 100
            functional_pcts.append(func_pct)
        
        # Sort by locus
        sorted_idx = np.argsort(loci)
        loci = [loci[i] for i in sorted_idx]
        exact_pcts = [exact_pcts[i] for i in sorted_idx]
        functional_pcts = [functional_pcts[i] for i in sorted_idx]
        
        # Plot
        ax = axes[0]
        x = np.arange(len(loci))
        width = 0.35
        
        ax.bar(x - width/2, exact_pcts, width, label='Exact Matches')
        ax.bar(x + width/2, functional_pcts, width, label='All Matches')
        
        ax.set_title('Matching Percentage by Locus')
        ax.set_ylabel('Match %')
        ax.set_ylim(0, 105)  # Slightly above 100% for visibility
        ax.set_xticks(x)
        ax.set_xticklabels(loci)
        ax.legend()
        
        # Add percentage labels
        for i, v in enumerate(exact_pcts):
            ax.text(i - width/2, v + 2, f"{v:.0f}%", ha='center')
        for i, v in enumerate(functional_pcts):
            ax.text(i + width/2, v + 2, f"{v:.0f}%", ha='center')
        
        # 2. Average similarity by locus
        ax = axes[1]
        similarities = [results['locus_results'][locus]['average_similarity'] for locus in loci]
        
        bars = ax.bar(loci, similarities, color=sns.color_palette("viridis", len(loci)))
        
        ax.set_title('Average Similarity by Locus')
        ax.set_ylabel('Similarity (0-1)')
        ax.set_ylim(0, 1.05)
        
        # Add similarity values
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f"{height:.2f}", ha='center', va='bottom')
        
        # Add horizontal line for similarity threshold
        ax.axhline(y=self.similarity_threshold, color='r', linestyle='--', 
                  label=f'Similarity Threshold ({self.similarity_threshold})')
        ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        if output_file:
            try:
                logger.info(f"Attempting to save figure to {output_file}")
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Successfully saved figure to {output_file}")
                plt.close(fig)
                return None
            except Exception as e:
                logger.error(f"Error saving figure to {output_file}: {e}")
                plt.close(fig)
                return fig
        else:
            return fig


class DonorRankingAnalyzer:
    """Ranks potential donors based on HLA matching with a recipient"""
    
    def __init__(
        self, 
        encoder,
        matching_analyzer=None,
        weights=None
    ):
        """Initialize donor ranking analyzer
        
        Args:
            encoder: HLAEncoder instance
            matching_analyzer: Optional custom MatchingAnalyzer
            weights: Dict with custom weights for different metrics
        """
        self.encoder = encoder
        self.matching_analyzer = matching_analyzer or MatchingAnalyzer(encoder)
        
        # Default weights for ranking metrics
        self.weights = {
            'exact_match_pct': 0.6,
            'functional_match_pct': 0.3,
            'average_similarity': 0.1,
            # Locus-specific weights come from matching_analyzer
        }
        
        # Update with custom weights if provided
        if weights:
            self.weights.update(weights)
    
    def rank_donors(self, recipient_alleles: List[str], donor_list: List[dict]) -> List[dict]:
        """Rank potential donors based on HLA matching
        
        Args:
            recipient_alleles: List of recipient HLA alleles
            donor_list: List of donor dicts, each with 'id' and 'alleles' keys
            
        Returns:
            List of donor dicts sorted by ranking, with added matching metrics
        """
        ranked_donors = []
        
        for donor in donor_list:
            donor_id = donor['id']
            donor_alleles = donor['alleles']
            
            # Analyze matching
            results = self.matching_analyzer.analyze_matching(donor_alleles, recipient_alleles)
            
            # Calculate ranking score
            score = (
                results['exact_match_pct'] * self.weights['exact_match_pct'] +
                results['functional_match_pct'] * self.weights['functional_match_pct'] +
                results['average_similarity'] * self.weights['average_similarity']
            )
            
            # Create donor result with matching data
            donor_result = {
                'id': donor_id,
                'score': score,
                'exact_match_pct': results['exact_match_pct'],
                'functional_match_pct': results['functional_match_pct'],
                'average_similarity': results['average_similarity'],
                'exact_matches': len(results['exact_matches']),
                'functional_matches': len(results['functional_matches']),
                'locus_results': results['locus_results'],
                'common_loci': results['common_loci'],
                'alleles': donor_alleles  # Include original alleles
            }
            
            ranked_donors.append(donor_result)
        
        # Sort by score (descending)
        ranked_donors.sort(key=lambda x: x['score'], reverse=True)
        
        # Add rank
        for i, donor in enumerate(ranked_donors):
            donor['rank'] = i + 1
            
        return ranked_donors
    
    def generate_ranking_report(
        self, 
        ranked_donors: List[dict],
        top_n: int = None,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate a donor ranking report
        
        Args:
            ranked_donors: List of ranked donor dicts from rank_donors
            top_n: Number of top donors to include (None for all)
            output_file: Path to save CSV report
            
        Returns:
            DataFrame with ranking report
        """
        # Limit to top N if specified
        if top_n:
            donors = ranked_donors[:top_n]
        else:
            donors = ranked_donors
        
        # Create report rows
        rows = []
        for donor in donors:
            row = {
                'Rank': donor['rank'],
                'Donor ID': donor['id'],
                'Score': f"{donor['score']:.4f}",
                'Exact Match %': f"{donor['exact_match_pct']*100:.1f}%",
                'Functional Match %': f"{donor['functional_match_pct']*100:.1f}%",
                'Average Similarity': f"{donor['average_similarity']:.3f}",
                'Common Loci': ', '.join(donor['common_loci'])
            }
            rows.append(row)
        
        # Create DataFrame
        report_df = pd.DataFrame(rows)
        
        # Save to file if specified
        if output_file:
            report_df.to_csv(output_file, index=False)
            logger.info(f"Ranking report saved to {output_file}")
        
        return report_df
    
    def visualize_ranking(
        self, 
        ranked_donors: List[dict],
        top_n: int = 10,
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """Visualize donor ranking
        
        Args:
            ranked_donors: List of ranked donor dicts from rank_donors
            top_n: Number of top donors to visualize
            output_file: Path to save visualization
            figsize: Figure size
            
        Returns:
            Matplotlib figure if output_file is None, otherwise None
        """
        if not PLOTTING_AVAILABLE:
            logger.error("Matplotlib not installed; cannot create visualization")
            return None
            
        # Limit to top N
        donors = ranked_donors[:top_n]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # 1. Overall scores
        ax = axes[0]
        
        donor_ids = [str(donor['id']) for donor in donors]
        scores = [donor['score'] for donor in donors]
        exact_pcts = [donor['exact_match_pct'] for donor in donors]
        func_pcts = [donor['functional_match_pct'] for donor in donors]
        similarities = [donor['average_similarity'] for donor in donors]
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'Donor ID': donor_ids,
            'Overall Score': scores,
            'Exact Match %': exact_pcts,
            'Functional Match %': func_pcts,
            'Average Similarity': similarities
        })
        
        # Normalize for comparison
        plot_df['Exact Match %'] = plot_df['Exact Match %'] / max(plot_df['Exact Match %']) if max(plot_df['Exact Match %']) > 0 else 0
        plot_df['Functional Match %'] = plot_df['Functional Match %'] / max(plot_df['Functional Match %']) if max(plot_df['Functional Match %']) > 0 else 0
        plot_df['Average Similarity'] = plot_df['Average Similarity'] / max(plot_df['Average Similarity']) if max(plot_df['Average Similarity']) > 0 else 0
        
        # Plot
        x = np.arange(len(donor_ids))
        width = 0.2
        
        ax.bar(x - width, plot_df['Overall Score'], width, label='Overall Score', color='blue')
        ax.bar(x, plot_df['Exact Match %'], width, label='Exact Match %', color='green')
        ax.bar(x + width, plot_df['Functional Match %'], width, label='Functional Match %', color='orange')
        
        ax.set_title('Donor Ranking Scores')
        ax.set_xticks(x)
        ax.set_xticklabels(donor_ids, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.05)
        
        # 2. Locus-specific matching
        ax = axes[1]
        
        # Get common loci across all donors
        common_loci = set()
        for donor in donors:
            common_loci.update(donor['common_loci'])
        common_loci = sorted(list(common_loci))
        
        # Create matrix of locus matches
        match_matrix = np.zeros((len(donors), len(common_loci)))
        
        for i, donor in enumerate(donors):
            for j, locus in enumerate(common_loci):
                if locus in donor['locus_results']:
                    locus_data = donor['locus_results'][locus]
                    total = locus_data['total_alleles']
                    if total > 0:
                        match_pct = (locus_data['exact_matches'] + locus_data['functional_matches']) / total
                        match_matrix[i, j] = match_pct
        
        # Create heatmap
        im = ax.imshow(match_matrix, cmap='viridis', aspect='auto')
        
        # Add labels
        ax.set_xticks(np.arange(len(common_loci)))
        ax.set_yticks(np.arange(len(donors)))
        ax.set_xticklabels(common_loci)
        ax.set_yticklabels(donor_ids)
        ax.set_title('Locus-Specific Matching Percentage')
        ax.set_ylabel('Donor ID')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Match Percentage')
        
        # Add text annotations
        for i in range(len(donors)):
            for j in range(len(common_loci)):
                text = ax.text(j, i, f"{match_matrix[i, j]:.2f}",
                              ha="center", va="center", color="w" if match_matrix[i, j] < 0.7 else "black")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        if output_file:
            try:
                logger.info(f"Attempting to save figure to {output_file}")
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Successfully saved figure to {output_file}")
                plt.close(fig)
                return None
            except Exception as e:
                logger.error(f"Error saving figure to {output_file}: {e}")
                plt.close(fig)
                return fig
        else:
            return fig
