# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that combines document retrieval with language generation to provide accurate, context-aware responses.

## Features

- **Document Processing**: Upload and process PDF documents
- **Vector Database**: Uses ChromaDB for efficient document storage and retrieval
- **Intelligent Retrieval**: Finds relevant document chunks based on user queries
- **Response Generation**: Generates contextual responses using retrieved information
- **Interactive Interface**: Easy-to-use chat interface

## Technologies Used

- **Python**: Core programming language
- **LangChain**: Framework for building LLM applications
- **ChromaDB**: Vector database for document storage
- **PDF Processing**: Document parsing and text extraction
- **Vector Embeddings**: For semantic search and retrieval

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aayushyatiwari/RAGfinal.git
cd RAGfinal
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

## Usage

1. **Start the application**:
```bash
python transcipt_1_agent.py
```

2. **Upload documents**: Add PDF file location of transcipt_1

3. **Ask questions**: Query the chatbot about the uploaded documents

4. **Get responses**: Receive accurate, context-aware answers based on your documents

## Project Structure

```
RAG-chatbot/
├── generation.py           # Response generation logic
├── chroma_langchain_db/   # ChromaDB vector database
├── transcipt_1.pdf       # Sample document
├── transcipt_1_agent.py  # Agent processing script
├── BFS_Share_Price.csv   # Sample data file
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── .env                 # Environment variables (not tracked)
```

## Configuration

Edit the `.env` file to configure:
- API keys for language models
- Database connection settings
- Model parameters
- Other application settings

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- **Author**: Aayush Yatiwari
- **GitHub**: [@aayushyatiwari](https://github.com/aayushyatiwari)
- **Repository**: [RAG-chatbot](https://github.com/aayushyatiwari/RAG-chatbot)

## Acknowledgments

- LangChain community for the excellent framework
- ChromaDB for the vector database solution
- OpenAI for language model capabilities
