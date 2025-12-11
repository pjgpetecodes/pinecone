"""
Query User Facial Similarity
Interactive interface to find users most similar by facial features.
Uses Pinecone vector search to measure and rank embedding distances.

EDUCATIONAL PURPOSE: Demonstrates vector similarity search fundamentals.
Educational context: Teaching how embeddings enable similarity measurement.
"""

import os
import json
import webbrowser
from datetime import datetime
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from dotenv import load_dotenv
from face_embedding_helper import FaceEmbeddingHelper

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", "pclocal"))
index_name = "user-faces-index"

if not pc.has_index(index_name):
    print(f"Error: Index '{index_name}' does not exist.")
    print(f"Run 'python index_user_faces.py' first to create the index.")
    exit(1)

index = pc.Index(index_name)

# Load users
with open("data/store/users.json", "r") as f:
    users = json.load(f)

user_map = {user['id']: user for user in users}


class FacialSimilarityEngine:
    """Search for users with similar facial characteristics."""
    
    def __init__(self):
        self.face_helper = FaceEmbeddingHelper()
    
    def query_by_user_id(self, user_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find users most similar to a given user's facial features.
        
        Args:
            user_id: User ID to find similar users for (e.g., "USER-001")
            top_k: Number of similar users to return
            
        Returns:
            List of similar users ranked by facial similarity
        """
        user = user_map.get(user_id)
        if not user:
            print(f"User {user_id} not found")
            return []
        
        profile_image_url = user.get('profile_image_url')
        if not profile_image_url:
            print(f"User {user_id} has no profile image")
            return []
        
        # Get user's facial embedding
        user_embedding = self.face_helper.get_face_embedding(profile_image_url, user_id)
        if user_embedding is None:
            print(f"Could not extract facial embedding for {user_id}")
            return []
        
        # Query Pinecone for similar faces
        results = index.query(
            vector=user_embedding,
            top_k=top_k + 1,  # +1 to exclude self
            include_metadata=True
        )
        
        # Format and filter results (exclude the user themselves)
        similar_users = []
        for i, result in enumerate(results['matches']):
            # Skip self match
            if result['id'] == f"{user_id}-face":
                continue
            
            metadata = result['metadata']
            similar_users.append({
                'user_id': metadata['user_id'],
                'name': metadata['name'],
                'segment': metadata['segment'],
                'profile_image_url': metadata['profile_image_url'],
                'similarity_score': result['score'],
                'rank': len(similar_users) + 1
            })
            
            if len(similar_users) >= top_k:
                break
        
        # Attach local cached image path hints
        for item in similar_users:
            item['cached_image_path'] = os.path.join('data', 'store', 'user_images', f"{item['user_id']}.jpg")
        return similar_users
    
    def query_by_image(self, image_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find users with facial features similar to a provided image.
        
        Images are automatically normalized to 256×256 (aspect-preserving center-crop)
        to match the preprocessing used during indexing, ensuring consistent embeddings.
        
        Args:
            image_path: Local file path or URL to the image
            top_k: Number of similar users to return
            
        Returns:
            List of similar users ranked by facial similarity
        """
        # Prefer cached file if URL matches a known user's profile image (for consistent preprocessing)
        resolved_source = image_path
        try:
            if image_path.startswith('http'):
                base = image_path.split('?')[0]
                for u in users:
                    u_url = u.get('profile_image_url', '')
                    if u_url:
                        u_base = u_url.split('?')[0]
                        if base == u_base:
                            # Use cached local file for embedding consistency
                            resolved_source = os.path.join('data', 'store', 'user_images', f"{u['id']}.jpg")
                            print(f"Resolved URL to cached file for consistent embedding: {resolved_source}")
                            break
        except Exception:
            pass

        # Extract embedding from resolved image source (normalizes to 256×256 before embedding)
        print(f"Generating embedding with normalized image preprocessing (256×256)...")
        face_embedding = self.face_helper.get_face_embedding(resolved_source)
        if face_embedding is None:
            print(f"Could not extract facial embedding from {resolved_source}")
            return []
        
        # Query Pinecone
        results = index.query(
            vector=face_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        similar_users = []
        for i, result in enumerate(results['matches']):
            metadata = result['metadata']
            similar_users.append({
                'user_id': metadata['user_id'],
                'name': metadata['name'],
                'segment': metadata['segment'],
                'profile_image_url': metadata['profile_image_url'],
                'similarity_score': result['score'],
                'rank': i + 1
            })
        
        # Attach local cached image path hints
        for item in similar_users:
            item['cached_image_path'] = os.path.join('data', 'store', 'user_images', f"{item['user_id']}.jpg")
        return similar_users


def display_results(results: List[Dict[str, Any]], query_type: str = "facial similarity", query_context: Optional[Dict[str, Any]] = None):
    """Display similarity results in formatted table and offer HTML gallery view."""
    if not results:
        print(f"\nNo similar users found for your {query_type} query.")
        return
    
    print(f"\n{'─' * 100}")
    print(f"Top Results - {query_type.upper()}")
    print(f"{'─' * 100}")
    
    for result in results:
        print(f"\n{result['rank']}. {result['name']} ({result['user_id']})")
        print(f"   Segment: {result['segment']}")
        print(f"   Facial Similarity Score: {result['similarity_score']:.4f}")
        if result['profile_image_url']:
            print(f"   Profile: {result['profile_image_url']}")
        if result.get('cached_image_path') and os.path.exists(result['cached_image_path']):
            print(f"   Cached Image: {result['cached_image_path']}")

    # Offer to open an HTML gallery of the results
    try:
        open_gallery = input("\nOpen image gallery in browser? (y/N): ").strip().lower() == 'y'
    except Exception:
        open_gallery = False
    if open_gallery:
        html_path = render_results_gallery(results, query_type, query_context)
        if html_path:
            print(f"\nOpening gallery: {html_path}")
            try:
                webbrowser.open(f"file://{os.path.abspath(html_path)}")
            except Exception:
                print("Could not automatically open browser. Please open the HTML file manually.")


def render_results_gallery(results: List[Dict[str, Any]], query_type: str, query_context: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Generate a simple HTML gallery showing cached images with similarity scores."""
    # Ensure reports directory exists
    reports_dir = os.path.join('data', 'store', 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    html_file = os.path.join(reports_dir, f"face_similarity_{timestamp}.html")

    # Derive query subject image src if provided
    query_subject_block = ""
    if query_context:
        # If an explicit subject source was passed, use it exactly.
        explicit_src = query_context.get('subject_src')
        subject_img_src = ""
        cached_path = query_context.get('cached_image_path')
        profile_url = query_context.get('profile_image_url')
        image_path = query_context.get('image_path')
        image_url = query_context.get('image_url')

        if explicit_src:
            subject_img_src = explicit_src
        else:
            # Priority: explicit URL/file input -> cached path -> profile URL
            if image_url and (image_url.startswith('http://') or image_url.startswith('https://') or image_url.startswith('file://')):
                subject_img_src = image_url
            elif image_path:
                if os.path.isabs(image_path) and os.path.exists(image_path):
                    subject_img_src = "file://" + os.path.abspath(image_path).replace('\\', '/')
                elif os.path.exists(image_path):
                    subject_img_src = os.path.relpath(image_path, start=reports_dir).replace('\\', '/')
                else:
                    subject_img_src = image_path
            elif cached_path and os.path.exists(cached_path):
                subject_img_src = os.path.relpath(cached_path, start=reports_dir).replace('\\', '/')
            elif profile_url:
                subject_img_src = profile_url

        subject_title = query_context.get('name') or query_context.get('title') or 'Selected Subject'
        subject_id = query_context.get('user_id', '')
        subject_segment = query_context.get('segment', '')
        if subject_img_src:
            query_subject_block = f"""
            <div class='subject'>
                <img src='{subject_img_src}' alt='{subject_title}'/>
                <div class='meta'>
                    <div class='name'>{subject_title} {('(' + subject_id + ')') if subject_id else ''}</div>
                    <div class='segment'>{subject_segment}</div>
                </div>
            </div>
            """

    # Build HTML content
    rows = []
    for r in results:
        img_path = r.get('cached_image_path')
        # Use local cached image if available, otherwise fall back to remote URL
        if img_path and os.path.exists(img_path):
            # Make path relative to the report directory to avoid nested duplication
            rel_src = os.path.relpath(img_path, start=reports_dir).replace('\\', '/')
            img_src = rel_src
        else:
            img_src = r.get('profile_image_url', '')
        rows.append(f"""
        <div class='card'>
            <img src='{img_src}' alt='{r.get('name','')}'/>
            <div class='meta'>
            <div class='name'>{r.get('rank','.')}. {r.get('name','')} ({r.get('user_id','')})</div>
            <div class='segment'>{r.get('segment','')}</div>
            <div class='score'>Similarity: {r.get('similarity_score',0):.4f}</div>
            </div>
        </div>
        """)

    html = f"""
        <!doctype html>
        <html>
            <head>
                <meta charset='utf-8'>
                <title>Facial Similarity Results</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ margin-bottom: 10px; }}
                    .subject {{ display: flex; gap: 16px; align-items: center; border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
                    .subject img {{ width: 140px; height: 140px; object-fit: cover; border-radius: 6px; }}
                    .subject .meta {{ display: flex; flex-direction: column; }}
                    .subject .name {{ font-weight: 700; font-size: 16px; margin-bottom: 6px; }}
                    .subject .segment {{ color: #555; font-size: 12px; }}
                    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; }}
                    .card {{ border: 1px solid #ddd; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
                    .card img {{ width: 100%; height: 220px; object-fit: cover; display: block; }}
                    .meta {{ padding: 10px; }}
                    .name {{ font-weight: 600; margin-bottom: 6px; }}
                    .segment {{ color: #555; font-size: 12px; margin-bottom: 4px; }}
                    .score {{ color: #0a7; font-size: 13px; font-weight: 600; }}
                    .note {{ margin-top: 16px; color: #666; font-size: 12px; }}
                </style>
            </head>
            <body>
                <h1>Facial Similarity Results ({query_type})</h1>
                {query_subject_block}
                <div class='grid'>
                    {''.join(rows)}
                </div>
                <div class='note'>Images use cached local files when available (data/store/user_images/), otherwise remote URLs.</div>
            </body>
        </html>
        """

    try:
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html)
        return html_file
    except Exception as e:
        print(f"Failed to write gallery: {e}")
        return None


def run_interactive_similarity():
    """Interactive facial similarity search interface."""
    print("\n" + "="*100)
    print("FACIAL SIMILARITY SEARCH".center(100))
    print("Find users with similar facial characteristics".center(100))
    print("="*100)
    print("\n⚠️  EDUCATIONAL PURPOSE: This demonstrates vector similarity measurement.")
    print("   It measures facial embedding distance for learning vector search concepts.")
    
    engine = FacialSimilarityEngine()
    
    while True:
        print("\n\nSearch Options:")
        print("1. Find users similar to a specific user (by ID)")
        print("2. Find users similar to an image (upload or provide path)")
        print("3. View all users")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            user_id = input("Enter user ID (e.g., USER-001): ").strip().upper()
            if user_id in user_map:
                print(f"\nSearching for users similar to {user_map[user_id]['name']}...")
                results = engine.query_by_user_id(user_id, top_k=5)
                # Build query context for selected user card
                query_context = {
                    'subject_type': 'user',
                    'user_id': user_id,
                    'name': user_map[user_id]['name'],
                    'segment': user_map[user_id]['segment'],
                    'profile_image_url': user_map[user_id].get('profile_image_url', ''),
                    'cached_image_path': os.path.join('data', 'store', 'user_images', f"{user_id}.jpg")
                }
                display_results(results, "facial similarity to user", query_context)
            else:
                print(f"User {user_id} not found")
        
        elif choice == "2":
            image_path = input("Enter image path or URL: ").strip()
            if image_path:
                print("\nAnalyzing facial features and searching...")
                results = engine.query_by_image(image_path, top_k=5)
                query_context = {
                    'subject_type': 'image',
                    'title': 'Selected Image',
                    # Pass exact input through both fields; renderer will choose
                    'image_path': image_path if not image_path.startswith(('http://','https://','file://')) else '',
                    'image_url': image_path if image_path.startswith(('http://','https://','file://')) else '',
                    'profile_image_url': '',
                    # Force exact use of the provided input in the subject block
                    'subject_src': image_path if image_path else ''
                }
                display_results(results, "facial similarity to image", query_context)
        
        elif choice == "3":
            print("\n" + "─"*100)
            print("All Users".center(100))
            print("─"*100)
            for user in users:
                print(f"\n{user['id']}: {user['name']}")
                print(f"  Segment: {user['segment']}")
                print(f"  Preferred Categories: {', '.join(user['preferred_categories'])}")
        
        elif choice == "4":
            print("\nExiting facial similarity search. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    try:
        run_interactive_similarity()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
