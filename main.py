import streamlit as st
import os
import tempfile
import json
import requests
import time
import sqlite3
import subprocess
import sys
import hashlib
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# Install faiss-cpu for Windows compatibility
try:
    import faiss
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
    import faiss

# Install LangChain components if not available
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.docstore.in_memory import InMemoryDocstore
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain langchain-community"])
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.docstore.in_memory import InMemoryDocstore

# Configuration
class Config:
    # Model settings
    EMBEDDING_MODEL_LOCAL = "BAAI/bge-small-en-v1.5"
    DEFAULT_OLLAMA_MODELS = ["llama3:8b", "mistral", "phi3", "gemma:2b", "nomic-embed-text"]
    TEMPERATURE = 0.7
    MAX_TOKENS = 4096
    
    # Knowledge base settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    SUPPORTED_FILE_TYPES = ["pdf", "txt", "md"]
    
    # Database settings
    DB_FILE = "culinary_sessions.db"
    
    # Ollama settings
    OLLAMA_URL = "http://localhost:11434"

# Data Models
@dataclass
class Recipe:
    id: str
    name: str
    description: str
    prep_time: int
    cook_time: int
    servings: int
    ingredients: List[Dict[str, str]]
    instructions: List[str]
    nutritional_info: Dict[str, Any]
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'prep_time': self.prep_time,
            'cook_time': self.cook_time,
            'servings': self.servings,
            'ingredients': self.ingredients,
            'instructions': self.instructions,
            'nutritional_info': self.nutritional_info,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Recipe':
        return cls(**data)

@dataclass
class Session:
    id: str
    name: str
    created_at: str
    updated_at: str
    recipes: List[Recipe]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'recipes': [recipe.to_dict() for recipe in self.recipes]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        recipes = [Recipe.from_dict(recipe) for recipe in data.get('recipes', [])]
        return cls(
            id=data['id'],
            name=data['name'],
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            recipes=recipes
        )

def generate_id() -> str:
    return hashlib.md5(f"{time.time()}{os.urandom(8)}".encode()).hexdigest()[:8]

# Database Operations
class Database:
    def __init__(self, db_file: str):
        self.db_file = db_file
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            data TEXT NOT NULL
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_session(self, session: Session) -> bool:
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        session.updated_at = datetime.now().isoformat()
        session_data = json.dumps(session.to_dict())
        
        cursor.execute('''
        INSERT OR REPLACE INTO sessions (id, name, created_at, updated_at, data)
        VALUES (?, ?, ?, ?, ?)
        ''', (session.id, session.name, session.created_at, session.updated_at, session_data))
        
        conn.commit()
        conn.close()
        return True
    
    def load_session(self, session_id: str) -> Optional[Session]:
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('SELECT data FROM sessions WHERE id = ?', (session_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            try:
                session_data = json.loads(result[0])
                return Session.from_dict(session_data)
            except Exception:
                return None
        
        return None
    
    def delete_session(self, session_id: str) -> bool:
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
        affected_rows = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return affected_rows > 0
    
    def search_sessions(self, query: str) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, name, created_at, updated_at FROM sessions
        WHERE name LIKE ?
        ORDER BY updated_at DESC
        ''', (f'%{query}%',))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'name': row[1],
                'created_at': row[2],
                'updated_at': row[3]
            })
        
        conn.close()
        return results
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, name, created_at, updated_at FROM sessions
        ORDER BY updated_at DESC
        ''')
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'name': row[1],
                'created_at': row[2],
                'updated_at': row[3]
            })
        
        conn.close()
        return results

# Knowledge Base Operations
class KnowledgeBase:
    def __init__(self):
        self.vector_store = None
        self.document_cache = {}
    
    def initialize_embeddings(self, model_name: str = Config.EMBEDDING_MODEL_LOCAL):
        if model_name in self.document_cache:
            return self.document_cache[model_name]
        
        try:
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            self.document_cache[model_name] = embeddings
            return embeddings
        except Exception:
            if Config.EMBEDDING_MODEL_LOCAL in self.document_cache:
                return self.document_cache[Config.EMBEDDING_MODEL_LOCAL]
            else:
                embeddings = HuggingFaceBgeEmbeddings(model_name=Config.EMBEDDING_MODEL_LOCAL)
                self.document_cache[Config.EMBEDDING_MODEL_LOCAL] = embeddings
                return embeddings
    
    def process_documents(self, documents, embedding_model: str):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        embeddings = self.initialize_embeddings(embedding_model)
        
        try:
            test_embedding = embeddings.embed_query("test")
            dimension = len(test_embedding)
            index = faiss.IndexFlatL2(dimension)
            
            vector_store = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={}
            )
            
            vector_store.add_documents(chunks)
            self.vector_store = vector_store
            return vector_store
        except Exception:
            return None
    
    def load_uploaded_files(self, uploaded_files):
        documents = []
        for file in uploaded_files:
            suffix = Path(file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name
            
            try:
                if file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path)
                
                docs = loader.load()
                documents.extend(docs)
            except Exception:
                pass
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        return documents
    
    def search(self, query: str, k: int = 3) -> List[str]:
        if not self.vector_store:
            return []
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception:
            return []

# Recipe Generation Logic
class RecipeGenerator:
    def __init__(self, ollama_url: str = Config.OLLAMA_URL):
        self.ollama_url = ollama_url
    
    def generate_with_ollama(self, prompt: str, model_name: str, kb_context: str = "") -> Optional[str]:
        try:
            # Enhanced prompt for recipe generation
            enhanced_prompt = f"""
            You are a professional chef. Create a detailed recipe based on the following request.
            Include ingredients with precise measurements, step-by-step instructions, prep time, cook time, and servings.
            Also provide basic nutritional information (calories, protein, carbs, fat).
            
            {kb_context}
            
            User Request: {prompt}
            
            Recipe:
            """
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": enhanced_prompt,
                    "stream": False,
                    "options": {
                        "temperature": Config.TEMPERATURE,
                        "num_predict": Config.MAX_TOKENS
                    }
                }
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            return None
        except Exception:
            return None
    
    def parse_from_text(self, text: str) -> Optional[Recipe]:
        try:
            # Initialize recipe with basic info
            recipe_id = generate_id()
            now = datetime.now().isoformat()
            
            # Extract title
            title_match = re.search(r'(?:title|name)[:\s]+([^\n]+)', text, re.IGNORECASE)
            name = title_match.group(1).strip() if title_match else "Untitled Recipe"
            
            # Extract description
            desc_match = re.search(r'(?:description|about)[:\s]+([^\n]+(?:\n[^#\n]+)*)', text, re.IGNORECASE)
            description = desc_match.group(1).strip() if desc_match else ""
            
            # Extract prep time
            prep_match = re.search(r'(?:prep|preparation)(?:\s+time)?[:\s]+(\d+)(?:\s+minutes?)?', text, re.IGNORECASE)
            prep_time = int(prep_match.group(1)) if prep_match else 0
            
            # Extract cook time
            cook_match = re.search(r'(?:cook|cooking)(?:\s+time)?[:\s]+(\d+)(?:\s+minutes?)?', text, re.IGNORECASE)
            cook_time = int(cook_match.group(1)) if cook_match else 0
            
            # Extract servings
            servings_match = re.search(r'(?:servings|serves|yields?)[:\s]+(\d+)', text, re.IGNORECASE)
            servings = int(servings_match.group(1)) if servings_match else 1
            
            # Extract ingredients
            ingredients = []
            ingredients_section = re.search(r'(?:ingredients)[:\s\n]+([^\n#]+(?:\n[^#\n]+)*)', text, re.IGNORECASE)
            if ingredients_section:
                ingredient_lines = ingredients_section.group(1).strip().split('\n')
                for line in ingredient_lines:
                    if line.strip():
                        # Try to parse amount and name
                        amount_match = re.match(r'^([\d\/\s\.]+(?:\s+(?:cup|cups|tbsp|tsp|oz|lb|g|kg|ml|l)?)?)\s+(.+)$', line.strip())
                        if amount_match:
                            amount = amount_match.group(1).strip()
                            name = amount_match.group(2).strip()
                        else:
                            amount = ""
                            name = line.strip()
                        
                        ingredients.append({"name": name, "amount": amount})
            
            # Extract instructions
            instructions = []
            instructions_section = re.search(r'(?:instructions|directions|steps|method)[:\s\n]+([^\n#]+(?:\n[^#\n]+)*)', text, re.IGNORECASE)
            if instructions_section:
                instruction_lines = instructions_section.group(1).strip().split('\n')
                for line in instruction_lines:
                    if line.strip():
                        # Try to extract step number
                        step_num_match = re.match(r'^(\d+)[\.\)]\s*(.+)$', line.strip())
                        if step_num_match:
                            instruction = step_num_match.group(2).strip()
                        else:
                            instruction = line.strip()
                        
                        instructions.append(instruction)
            
            # Extract nutritional information
            nutritional_info = {}
            nutrition_section = re.search(r'(?:nutrition|nutritional\s+information)[:\s\n]+([^\n#]+(?:\n[^#\n]+)*)', text, re.IGNORECASE)
            if nutrition_section:
                nutrition_text = nutrition_section.group(1).strip()
                
                # Extract calories
                calories_match = re.search(r'(?:calories|energy)[:\s]+(\d+)', nutrition_text, re.IGNORECASE)
                if calories_match:
                    nutritional_info["calories"] = int(calories_match.group(1))
                
                # Extract protein
                protein_match = re.search(r'(?:protein)[:\s]+(\d+(?:\.\d+)?)\s*g', nutrition_text, re.IGNORECASE)
                if protein_match:
                    nutritional_info["protein"] = float(protein_match.group(1))
                
                # Extract carbs
                carbs_match = re.search(r'(?:carbs|carbohydrates)[:\s]+(\d+(?:\.\d+)?)\s*g', nutrition_text, re.IGNORECASE)
                if carbs_match:
                    nutritional_info["carbs"] = float(carbs_match.group(1))
                
                # Extract fat
                fat_match = re.search(r'(?:fat)[:\s]+(\d+(?:\.\d+)?)\s*g', nutrition_text, re.IGNORECASE)
                if fat_match:
                    nutritional_info["fat"] = float(fat_match.group(1))
            
            # Create recipe object
            recipe = Recipe(
                id=recipe_id,
                name=name,
                description=description,
                prep_time=prep_time,
                cook_time=cook_time,
                servings=servings,
                ingredients=ingredients,
                instructions=instructions,
                nutritional_info=nutritional_info,
                created_at=now
            )
            
            return recipe
        except Exception:
            return None

# UI Components
def check_ollama_connection(url: str) -> bool:
    try:
        response = requests.get(f"{url}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def fetch_ollama_models(url: str) -> List[str]:
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
    except Exception:
        pass
    return []

def render_sidebar(knowledge_base, current_session, db):
    with st.sidebar:
        # Header with app title and icon
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem; padding: 1rem; background-color: #f8f9fa; border-radius: 8px;">
            <h1 style="font-size: 1.5rem; margin: 0; color: #333;">🍳 Culinary Assistant</h1>
            <p style="font-size: 0.8rem; color: #666; margin: 0.5rem 0;">Your AI Recipe Generator</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Status indicators with better visual design
        ollama_connected = check_ollama_connection(Config.OLLAMA_URL)
        ollama_status = "🟢 Connected" if ollama_connected else "🔴 Disconnected"
        kb_status = "🟢 Active" if knowledge_base.vector_store else "🔴 Not Set"
        
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            <h4 style="margin-bottom: 0.5rem; font-size: 0.9rem;">System Status</h4>
            <div style="display: flex; justify-content: space-between; padding: 0.5rem; background-color: #f8f9fa; border-radius: 8px;">
                <span style="font-size: 0.8rem;">Ollama</span>
                <span style="font-size: 0.8rem; font-weight: bold;">""" + ollama_status + """</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 0.5rem; background-color: #f8f9fa; border-radius: 8px; margin-top: 0.5rem;">
                <span style="font-size: 0.8rem;">Knowledge Base</span>
                <span style="font-size: 0.8rem; font-weight: bold;">""" + kb_status + """</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Selection with better label
        if not st.session_state.ollama_models:
            st.session_state.ollama_models = fetch_ollama_models(Config.OLLAMA_URL) or Config.DEFAULT_OLLAMA_MODELS
        
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            <h4 style="margin-bottom: 0.5rem; font-size: 0.9rem;">AI Model</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.selected_model = st.selectbox(
            "Select Model",
            st.session_state.ollama_models,
            index=0 if not st.session_state.selected_model else st.session_state.ollama_models.index(st.session_state.selected_model) if st.session_state.selected_model in st.session_state.ollama_models else 0,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Knowledge Base with better visual design
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            <h4 style="margin-bottom: 0.5rem; font-size: 0.9rem;">Knowledge Base</h4>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload culinary documents",
            accept_multiple_files=True,
            type=Config.SUPPORTED_FILE_TYPES,
            label_visibility="collapsed",
            help="Upload PDF, TXT, or MD files to enhance recipe generation"
        )
        
        if uploaded_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{len(uploaded_files)} file(s) selected")
            with col2:
                if st.button("Process", use_container_width=True, key="process_kb_files"):
                    with st.spinner("Processing documents..."):
                        documents = knowledge_base.load_uploaded_files(uploaded_files)
                        if documents:
                            vector_store = knowledge_base.process_documents(documents, Config.EMBEDDING_MODEL_LOCAL)
                            if vector_store:
                                st.success(f"✅ {len(documents)} files processed")
                                st.rerun()
                            else:
                                st.error("Failed to process documents")
        
        if knowledge_base.vector_store:
            st.caption(f"📊 {knowledge_base.vector_store.index.ntotal} knowledge chunks indexed")
        
        st.markdown("---")
        
        # Session Management with better visual design
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            <h4 style="margin-bottom: 0.5rem; font-size: 0.9rem;">Recipe Collections</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Create new session
        if st.button("➕ New Collection", use_container_width=True, key="new_session"):
            session_id = generate_id()
            new_session = Session(
                id=session_id,
                name=f"Collection {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                recipes=[]
            )
            db.save_session(new_session)
            st.session_state.current_session = new_session
            st.rerun()
        
        # List sessions
        sessions = db.get_all_sessions()
        if sessions:
            for session in sessions[:5]:  # Show last 5 sessions
                is_current = current_session and current_session.id == session['id']
                button_label = f"{'🔵 ' if is_current else ''}{session['name']}"
                
                if st.button(button_label, use_container_width=True, key=f"session_{session['id']}"):
                    loaded_session = db.load_session(session['id'])
                    if loaded_session:
                        st.session_state.current_session = loaded_session
                        st.rerun()

def render_workspace(current_session, knowledge_base, recipe_generator, db):
    # Header with current session info
    if current_session:
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h3 style="margin: 0; color: #333;">📚 {current_session.name}</h3>
            <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">
                {len(current_session.recipes)} recipe{'' if len(current_session.recipes) == 1 else 's'} in this collection
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Please create or select a recipe collection from the sidebar")
        return
    
    # Recipe generation section
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h3 style="margin: 0 0 1rem 0; color: #333;">Create New Recipe</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Recipe generation form
    with st.form("recipe_form"):
        prompt = st.text_area(
            "Describe the recipe you want to create...",
            placeholder="e.g., A vegetarian pasta dish with tomatoes and basil",
            height=100
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption("The AI will generate a complete recipe based on your description")
        with col2:
            generate_button = st.form_submit_button("Generate Recipe", use_container_width=True)
    
    if generate_button and prompt:
        if not st.session_state.selected_model:
            st.error("Please select a model in the sidebar")
            return
        
        with st.spinner("Generating recipe..."):
            # Get context from knowledge base if available
            kb_context = ""
            if knowledge_base.vector_store:
                kb_results = knowledge_base.search(prompt, k=2)
                if kb_results:
                    kb_context = "\n\nKnowledge Base Context:\n" + "\n".join(kb_results[:2])
            
            recipe_text = recipe_generator.generate_with_ollama(
                prompt, 
                st.session_state.selected_model, 
                kb_context
            )
            
            if recipe_text:
                recipe = recipe_generator.parse_from_text(recipe_text)
                if recipe:
                    current_session.recipes.append(recipe)
                    db.save_session(current_session)
                    st.success(f"✅ Recipe '{recipe.name}' created successfully!")
                    st.rerun()
                else:
                    st.error("Failed to parse the generated recipe. Please try again.")
            else:
                st.error("Failed to generate recipe. Please check your connection and try again.")
    
    # Display recipes section
    if current_session.recipes:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h3 style="margin: 0 0 1rem 0; color: #333;">Your Recipes</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for i, recipe in enumerate(current_session.recipes):
            with st.container():
                # Recipe card header
                col1, col2, col3 = st.columns([4, 1, 1])
                with col1:
                    st.markdown(f"#### 📜 {recipe.name}")
                with col2:
                    st.markdown(f"**Time:** {recipe.prep_time + recipe.cook_time} min")
                with col3:
                    if st.button("🗑️", use_container_width=True, key=f"delete_{recipe.id}"):
                        current_session.recipes.remove(recipe)
                        db.save_session(current_session)
                        st.rerun()
                
                # Recipe details in tabs
                recipe_tab1, recipe_tab2, recipe_tab3 = st.tabs(["Overview", "Ingredients", "Instructions"])
                
                with recipe_tab1:
                    st.markdown(f"**Description:** {recipe.description}")
                    
                    # Time and servings info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prep Time", f"{recipe.prep_time} min")
                    with col2:
                        st.metric("Cook Time", f"{recipe.cook_time} min")
                    with col3:
                        st.metric("Servings", recipe.servings)
                    
                    # Nutritional info
                    if recipe.nutritional_info:
                        st.markdown("**Nutritional Information:**")
                        nutrition_cols = st.columns(4)
                        with nutrition_cols[0]:
                            if "calories" in recipe.nutritional_info:
                                st.metric("Calories", recipe.nutritional_info["calories"])
                        with nutrition_cols[1]:
                            if "protein" in recipe.nutritional_info:
                                st.metric("Protein", f"{recipe.nutritional_info['protein']}g")
                        with nutrition_cols[2]:
                            if "carbs" in recipe.nutritional_info:
                                st.metric("Carbs", f"{recipe.nutritional_info['carbs']}g")
                        with nutrition_cols[3]:
                            if "fat" in recipe.nutritional_info:
                                st.metric("Fat", f"{recipe.nutritional_info['fat']}g")
                
                with recipe_tab2:
                    if recipe.ingredients:
                        for ingredient in recipe.ingredients:
                            st.markdown(f"- **{ingredient['amount']}** {ingredient['name']}")
                    else:
                        st.info("No ingredients listed")
                
                with recipe_tab3:
                    if recipe.instructions:
                        for i, instruction in enumerate(recipe.instructions):
                            st.markdown(f"{i+1}. {instruction}")
                    else:
                        st.info("No instructions listed")
                
                st.markdown("---")
    else:
        st.info("No recipes yet. Create your first recipe using the form above!")

def render_knowledge_base(knowledge_base):
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h3 style="margin: 0 0 1rem 0; color: #333;">Knowledge Base</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if knowledge_base.vector_store:
        st.markdown(f"📊 {knowledge_base.vector_store.index.ntotal} knowledge chunks indexed")
        
        # Search interface
        with st.form("kb_search_form"):
            query = st.text_input(
                "Search knowledge base...",
                placeholder="e.g., Italian pasta dishes",
                label_visibility="collapsed"
            )
            k_results = st.slider("Number of results", min_value=1, max_value=10, value=5)
            search_button = st.form_submit_button("Search")
        
        if search_button and query:
            with st.spinner("Searching..."):
                docs = knowledge_base.vector_store.similarity_search(query, k=k_results)
                
                if docs:
                    for i, doc in enumerate(docs):
                        with st.expander(f"Result {i+1}", expanded=False):
                            st.markdown(doc.page_content)
                            st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                else:
                    st.info("No results found")
    else:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h4 style="margin: 0 0 1rem 0; color: #333;">No Knowledge Base</h4>
            <p style="margin: 0; color: #666;">Upload culinary documents in the sidebar to create a knowledge base.</p>
        </div>
        """, unsafe_allow_html=True)

# Main Application
def init_session_state():
    defaults = {
        'ollama_models': [],
        'selected_model': "",
        'current_session': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    # Configure page
    st.set_page_config(
        page_title="Culinary Assistant",
        page_icon="🍳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Improved CSS for better UI
    st.markdown("""
    <style>
        .main-header {
            display: none;
        }
        .stSelectbox > div > div > select {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
        }
        .stTextInput > div > div > input {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
        }
        .stTextArea > div > div > textarea {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
        }
        .stButton > button {
            border-radius: 4px;
            font-weight: 500;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border-radius: 8px;
            font-weight: 500;
        }
        .stMetric {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 0.5rem;
            border-radius: 4px;
            text-align: center;
        }
        div[data-testid="stForm"] {
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 1rem;
            background-color: #f8f9fa;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Initialize components
    db = Database(Config.DB_FILE)
    knowledge_base = KnowledgeBase()
    recipe_generator = RecipeGenerator()
    
    # Load current session if not already loaded
    if not st.session_state.current_session:
        # Get the most recent session
        sessions = db.get_all_sessions()
        if sessions:
            st.session_state.current_session = db.load_session(sessions[0]['id'])
    
    # Render sidebar
    render_sidebar(knowledge_base, st.session_state.current_session, db)
    
    # Main content tabs
    tab1, tab2 = st.tabs(["🍳 Recipes", "🔍 Knowledge Base"])
    
    with tab1:
        render_workspace(st.session_state.current_session, knowledge_base, recipe_generator, db)
    
    with tab2:
        render_knowledge_base(knowledge_base)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background-color: #f8f9fa; border-radius: 8px;">
        <p style="margin: 0; color: #666; font-size: 0.8rem;">Culinary Assistant - Powered by Ollama & FAISS</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()