#!/usr/bin/env python3
"""
ResiBot 2.0 - Technical Prototype
Demonstrating RAG + MCP Agent Architecture for 90% Communication Automation

This prototype showcases:
1. Intent classification using sentence embeddings
2. MCP Agent framework with specialized handlers
3. RAG-based policy knowledge retrieval
4. Multi-language support (German/English)
5. Confidence-based escalation
6. Performance metrics and logging

Requirements:
pip install sentence-transformers torch numpy scikit-learn fastapi uvicorn python-multipart
"""

import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ResiBot")

# ============================================================================
# Data Models
# ============================================================================

class IntentType(Enum):
    STATUS_CHECK = "status_check"
    DOCUMENT_CHECK = "document_check"
    APPOINTMENT = "appointment"
    POLICY_QUESTION = "policy_question"
    GENERAL_INQUIRY = "general_inquiry"

@dataclass
class IntentResult:
    intent: IntentType
    confidence: float
    agent: str
    processing_time: float

@dataclass
class CaseData:
    case_id: str
    name: str
    status: str
    eta: str
    caseworker: str
    submitted_docs: List[str]
    missing_docs: List[str]
    appointment: Optional[str]
    last_updated: str

@dataclass
class AgentResponse:
    agent_name: str
    response: str
    data: Dict
    confidence: float
    processing_time: float

class ChatRequest(BaseModel):
    message: str
    case_id: Optional[str] = "12345"  # Default for demo
    language: Optional[str] = "auto"

class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float
    agent_used: str
    processing_time: float
    requires_escalation: bool
    system_logs: List[str]

# ============================================================================
# Mock Data Layer (Simulates AMIS Database)
# ============================================================================

class MockAMISDatabase:
    """Simulates the Ausländer-Modul Information System (AMIS)"""
    
    def __init__(self):
        self.cases = {
            "12345": CaseData(
                case_id="12345",
                name="Frau Sarah Müller",
                status="Under Review",
                eta="2024-03-15",
                caseworker="mueller@lea-berlin.de",
                submitted_docs=["passport_scan_front", "employment_contract", "health_insurance"],
                missing_docs=["passport_scan_back", "Meldebescheinigung"],
                appointment="2024-02-20 14:30",
                last_updated="2024-01-28"
            ),
            "67890": CaseData(
                case_id="67890",
                name="Mr. Ahmed Hassan",
                status="Approved",
                eta="2024-02-28",
                caseworker="schmidt@lea-berlin.de",
                submitted_docs=["passport_scan_front", "passport_scan_back", "Meldebescheinigung", 
                               "employment_contract", "criminal_record", "health_insurance"],
                missing_docs=[],
                appointment=None,
                last_updated="2024-01-25"
            ),
            "11111": CaseData(
                case_id="11111",
                name="Ms. Elena Popov",
                status="Additional Documents Required",
                eta="2024-04-01",
                caseworker="weber@lea-berlin.de",
                submitted_docs=["passport_scan_front", "passport_scan_back"],
                missing_docs=["employment_contract", "Meldebescheinigung", "health_insurance"],
                appointment="2024-02-15 10:00",
                last_updated="2024-01-30"
            )
        }
    
    def get_case(self, case_id: str) -> Optional[CaseData]:
        return self.cases.get(case_id)
    
    def get_available_appointments(self, days_ahead: int = 14) -> List[str]:
        """Simulate available appointment slots"""
        slots = []
        base_date = datetime.now() + timedelta(days=3)
        for i in range(days_ahead):
            date = base_date + timedelta(days=i)
            if date.weekday() < 5:  # Monday to Friday
                for hour in [9, 10, 11, 14, 15, 16]:
                    slots.append(f"{date.strftime('%Y-%m-%d')} {hour:02d}:00")
        return slots[:10]  # Return first 10 available slots

# ============================================================================
# Policy Knowledge Base (Simulates RAG Vector Database)
# ============================================================================

class PolicyKnowledgeBase:
    """Simulates RAG-indexed policy documents"""
    
    def __init__(self):
        self.policies = {
            "§18a AufenthG": {
                "title": "Aufenthaltserlaubnis für qualifizierte Geduldete zum Zweck der Beschäftigung",
                "summary": "Residence permit for qualified tolerated persons for employment purposes",
                "requirements": [
                    "Valid passport or passport substitute document",
                    "Employment contract or binding job offer",
                    "Proof of residence registration (Meldebescheinigung)",
                    "Clean criminal record certificate (Führungszeugnis)",
                    "Proof of health insurance coverage",
                    "Proof of sufficient living space",
                    "Financial security proof (salary confirmation or bank statements)"
                ],
                "processing_time": "4-8 weeks",
                "fee": "€100",
                "validity": "Initially 2 years, renewable",
                "restrictions": "Tied to specific employer initially"
            },
            "§81a AufenthG": {
                "title": "Verfahren bei Änderungen",
                "summary": "Procedures for changes in residence status",
                "requirements": [
                    "Written notification of changes within 2 weeks",
                    "Updated documentation for new circumstances",
                    "Fee payment if applicable"
                ],
                "processing_time": "2-4 weeks",
                "fee": "€50-100 depending on change type"
            },
            "Document Requirements": {
                "passport_scan_front": "Clear, colored scan of passport main page with photo",
                "passport_scan_back": "Scan of passport back page with entry stamps",
                "Meldebescheinigung": "Certificate of residence registration from local Bürgeramt",
                "employment_contract": "Signed employment contract with salary and job description",
                "health_insurance": "Proof of health insurance coverage in Germany",
                "criminal_record": "Criminal record certificate from home country (apostilled)"
            }
        }
    
    def search_policy(self, query: str) -> Dict:
        """Simulate vector similarity search for policy information"""
        query_lower = query.lower()
        
        # Simple keyword matching (in real implementation, this would be vector similarity)
        if "18a" in query_lower or "employment" in query_lower or "beschäftigung" in query_lower:
            return self.policies["§18a AufenthG"]
        elif "81a" in query_lower or "änderung" in query_lower or "change" in query_lower:
            return self.policies["§81a AufenthG"]
        elif any(doc in query_lower for doc in ["document", "dokument", "unterlagen", "papers"]):
            return self.policies["Document Requirements"]
        else:
            return {"title": "General Information", "summary": "Please specify your policy question for detailed guidance"}

# ============================================================================
# Intent Classification Engine
# ============================================================================

class IntentClassifier:
    """Simplified intent classification using sentence embeddings"""
    
    def __init__(self):
        logger.info("Loading sentence transformer model...")
        # Using a multilingual model for German/English support
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Training examples for each intent
        self.intent_examples = {
            IntentType.STATUS_CHECK: [
                "What is the status of my application?",
                "Wie ist der Stand meines Antrags?",
                "Status check please",
                "Wo steht mein Verfahren?",
                "Application status update",
                "Wann wird mein Antrag bearbeitet?"
            ],
            IntentType.DOCUMENT_CHECK: [
                "What documents do I need?",
                "Welche Unterlagen fehlen noch?",
                "Document requirements",
                "Missing papers",
                "Was muss ich noch einreichen?",
                "Required documents list",
                "What documents do I still need to submit?",
                "Which papers are missing?",
                "Missing documents list",
                "Fehlende Unterlagen"
            ],
            IntentType.APPOINTMENT: [
                "I need an appointment",
                "Ich brauche einen Termin",
                "Schedule meeting",
                "Appointment booking",
                "Wann ist mein nächster Termin?",
                "Available time slots"
            ],
            IntentType.POLICY_QUESTION: [
                "What are the requirements for §18a?",
                "Explain the law",
                "Policy question",
                "Legal requirements",
                "Was bedeutet §81a?",
                "Regulation explanation"
            ],
            IntentType.GENERAL_INQUIRY: [
                "General question",
                "Help me",
                "Information needed",
                "Allgemeine Frage",
                "I have a question",
                "Können Sie mir helfen?"
            ]
        }
        
        # Precompute embeddings for training examples
        self._compute_intent_embeddings()
        logger.info("Intent classifier initialized successfully")
    
    def _compute_intent_embeddings(self):
        """Precompute embeddings for all training examples"""
        self.intent_embeddings = {}
        for intent, examples in self.intent_examples.items():
            embeddings = self.model.encode(examples)
            self.intent_embeddings[intent] = embeddings
    
    def classify(self, text: str) -> IntentResult:
        """Classify user input intent with confidence score"""
        start_time = time.time()
        
        # Encode the input text
        input_embedding = self.model.encode([text])
        
        best_intent = IntentType.GENERAL_INQUIRY
        best_confidence = 0.0
        
        # Calculate similarity with each intent's examples
        for intent, embeddings in self.intent_embeddings.items():
            similarities = cosine_similarity(input_embedding, embeddings)[0]
            max_similarity = np.max(similarities)
            
            if max_similarity > best_confidence:
                best_confidence = max_similarity
                best_intent = intent
        
        # Map intent to agent
        agent_mapping = {
            IntentType.STATUS_CHECK: "status_agent",
            IntentType.DOCUMENT_CHECK: "doc_agent",
            IntentType.APPOINTMENT: "booking_agent",
            IntentType.POLICY_QUESTION: "policy_agent",
            IntentType.GENERAL_INQUIRY: "general_agent"
        }
        
        processing_time = time.time() - start_time
        
        return IntentResult(
            intent=best_intent,
            confidence=best_confidence,
            agent=agent_mapping[best_intent],
            processing_time=processing_time
        )

# ============================================================================
# MCP Agent Framework
# ============================================================================

class BaseAgent:
    """Base class for all MCP agents"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.call_count = 0
    
    async def process(self, query: str, case_id: str, context: Dict) -> AgentResponse:
        """Process a query and return response"""
        raise NotImplementedError
    
    def _format_response(self, message: str, data: Dict, confidence: float, start_time: float) -> AgentResponse:
        """Helper to format agent response"""
        return AgentResponse(
            agent_name=self.name,
            response=message,
            data=data,
            confidence=confidence,
            processing_time=time.time() - start_time
        )

class StatusAgent(BaseAgent):
    """Handles case status inquiries"""
    
    def __init__(self, amis_db: MockAMISDatabase):
        super().__init__("Status Agent", "Retrieves case status and timeline information")
        self.amis_db = amis_db
    
    async def process(self, query: str, case_id: str, context: Dict) -> AgentResponse:
        start_time = time.time()
        self.call_count += 1
        
        case = self.amis_db.get_case(case_id)
        if not case:
            return self._format_response(
                "I couldn't find a case with that ID. Please verify your case number.",
                {"error": "Case not found"},
                0.1,
                start_time
            )
        
        # Determine response language based on query
        is_german = any(word in query.lower() for word in ['status', 'stand', 'antrag', 'verfahren'])
        
        if is_german:
            response = f"""Guten Tag {case.name}!

Ihr Antragsstatus: {case.status}
Voraussichtliche Bearbeitung bis: {case.eta}
Zuständige Sachbearbeitung: {case.caseworker}
Letzte Aktualisierung: {case.last_updated}

Bei Rückfragen können Sie sich direkt an Ihre Sachbearbeitung wenden."""
        else:
            response = f"""Hello {case.name}!

Your application status: {case.status}
Expected processing completion: {case.eta}
Assigned caseworker: {case.caseworker}
Last updated: {case.last_updated}

For specific questions, please contact your assigned caseworker directly."""
        
        return self._format_response(
            response,
            asdict(case),
            0.95,
            start_time
        )

class DocumentAgent(BaseAgent):
    """Handles document requirement inquiries"""
    
    def __init__(self, amis_db: MockAMISDatabase, policy_kb: PolicyKnowledgeBase):
        super().__init__("Document Agent", "Analyzes document requirements and compliance")
        self.amis_db = amis_db
        self.policy_kb = policy_kb
    
    async def process(self, query: str, case_id: str, context: Dict) -> AgentResponse:
        start_time = time.time()
        self.call_count += 1
        
        case = self.amis_db.get_case(case_id)
        if not case:
            return self._format_response(
                "I couldn't find a case with that ID. Please verify your case number.",
                {"error": "Case not found"},
                0.1,
                start_time
            )
        
        is_german = any(word in query.lower() for word in ['dokument', 'unterlagen', 'fehlen', 'einreichen'])
        
        if case.missing_docs:
            if is_german:
                response = f"""Hallo {case.name}!

Noch fehlende Unterlagen:
"""
                for doc in case.missing_docs:
                    doc_info = self.policy_kb.policies["Document Requirements"].get(doc, doc)
                    response += f"• {doc}: {doc_info}\n"
                
                response += f"""
Bereits eingereicht: {', '.join(case.submitted_docs)}

Bitte reichen Sie die fehlenden Dokumente bis zum {case.eta} ein."""
            else:
                response = f"""Hello {case.name}!

Still missing documents:
"""
                for doc in case.missing_docs:
                    doc_info = self.policy_kb.policies["Document Requirements"].get(doc, doc)
                    response += f"• {doc}: {doc_info}\n"
                
                response += f"""
Already submitted: {', '.join(case.submitted_docs)}

Please submit the missing documents by {case.eta}."""
        else:
            if is_german:
                response = f"""Hallo {case.name}!

Alle erforderlichen Dokumente sind vollständig eingereicht:
{', '.join(case.submitted_docs)}

Ihr Antrag wird derzeit bearbeitet. Status: {case.status}"""
            else:
                response = f"""Hello {case.name}!

All required documents have been submitted:
{', '.join(case.submitted_docs)}

Your application is currently being processed. Status: {case.status}"""
        
        return self._format_response(
            response,
            {
                "missing_docs": case.missing_docs,
                "submitted_docs": case.submitted_docs,
                "case_status": case.status
            },
            0.92,
            start_time
        )

class BookingAgent(BaseAgent):
    """Handles appointment scheduling"""
    
    def __init__(self, amis_db: MockAMISDatabase):
        super().__init__("Booking Agent", "Manages appointment scheduling and availability")
        self.amis_db = amis_db
    
    async def process(self, query: str, case_id: str, context: Dict) -> AgentResponse:
        start_time = time.time()
        self.call_count += 1
        
        case = self.amis_db.get_case(case_id)
        if not case:
            return self._format_response(
                "I couldn't find a case with that ID. Please verify your case number.",
                {"error": "Case not found"},
                0.1,
                start_time
            )
        
        available_slots = self.amis_db.get_available_appointments()
        is_german = any(word in query.lower() for word in ['termin', 'appointment', 'meeting'])
        
        if case.appointment:
            if is_german:
                response = f"""Hallo {case.name}!

Ihr nächster Termin: {case.appointment}
Zuständige Sachbearbeitung: {case.caseworker}

Falls Sie den Termin verschieben möchten, hier sind verfügbare Alternativen:
"""
                for i, slot in enumerate(available_slots[:5], 1):
                    response += f"{i}. {slot}\n"
                
                response += "\nBitte kontaktieren Sie uns zur Terminänderung."
            else:
                response = f"""Hello {case.name}!

Your next appointment: {case.appointment}
Assigned caseworker: {case.caseworker}

If you need to reschedule, here are available alternatives:
"""
                for i, slot in enumerate(available_slots[:5], 1):
                    response += f"{i}. {slot}\n"
                
                response += "\nPlease contact us to reschedule your appointment."
        else:
            if is_german:
                response = f"""Hallo {case.name}!

Sie haben derzeit keinen Termin vereinbart.

Verfügbare Termine in den nächsten zwei Wochen:
"""
                for i, slot in enumerate(available_slots[:8], 1):
                    response += f"{i}. {slot}\n"
                
                response += f"\nZum Buchen kontaktieren Sie: {case.caseworker}"
            else:
                response = f"""Hello {case.name}!

You currently don't have a scheduled appointment.

Available slots in the next two weeks:
"""
                for i, slot in enumerate(available_slots[:8], 1):
                    response += f"{i}. {slot}\n"
                
                response += f"\nTo book, please contact: {case.caseworker}"
        
        return self._format_response(
            response,
            {
                "current_appointment": case.appointment,
                "available_slots": available_slots[:8],
                "caseworker": case.caseworker
            },
            0.89,
            start_time
        )

class PolicyAgent(BaseAgent):
    """Handles policy and regulatory questions"""
    
    def __init__(self, policy_kb: PolicyKnowledgeBase):
        super().__init__("Policy Agent", "Provides regulatory and legal guidance")
        self.policy_kb = policy_kb
    
    async def process(self, query: str, case_id: str, context: Dict) -> AgentResponse:
        start_time = time.time()
        self.call_count += 1
        
        policy_info = self.policy_kb.search_policy(query)
        is_german = any(word in query.lower() for word in ['was', 'wie', 'voraussetzung', 'bestimmung'])
        
        if "§18a" in query or "employment" in query.lower() or "beschäftigung" in query.lower():
            if is_german:
                response = f"""Informationen zu {policy_info['title']}:

Zusammenfassung: {policy_info['summary']}

Erforderliche Dokumente:
"""
                for req in policy_info['requirements']:
                    response += f"• {req}\n"
                
                response += f"""
Bearbeitungszeit: {policy_info['processing_time']}
Gebühr: {policy_info['fee']}
Gültigkeitsdauer: {policy_info['validity']}
Beschränkungen: {policy_info['restrictions']}"""
            else:
                response = f"""Information about {policy_info['title']}:

Summary: {policy_info['summary']}

Required documents:
"""
                for req in policy_info['requirements']:
                    response += f"• {req}\n"
                
                response += f"""
Processing time: {policy_info['processing_time']}
Fee: {policy_info['fee']}
Validity: {policy_info['validity']}
Restrictions: {policy_info['restrictions']}"""
        else:
            if is_german:
                response = f"""Allgemeine Politikinformationen:

{policy_info.get('summary', 'Bitte spezifizieren Sie Ihre Frage für detaillierte Informationen.')}

Für spezifische Fragen zu §18a AufenthG, §81a AufenthG oder anderen Bestimmungen, 
nennen Sie bitte die entsprechende Paragraphennummer."""
            else:
                response = f"""General policy information:

{policy_info.get('summary', 'Please specify your question for detailed guidance.')}

For specific questions about §18a AufenthG, §81a AufenthG, or other regulations,
please mention the specific paragraph number."""
        
        return self._format_response(
            response,
            policy_info,
            0.87,
            start_time
        )

# ============================================================================
# Main ResiBot System
# ============================================================================

class ResiBot:
    """Main ResiBot system orchestrating all components"""
    
    def __init__(self):
        logger.info("Initializing ResiBot 2.0 system...")
        
        # Initialize data layer
        self.amis_db = MockAMISDatabase()
        self.policy_kb = PolicyKnowledgeBase()
        
        # Initialize intent classifier
        self.intent_classifier = IntentClassifier()
        
        # Initialize MCP agents
        self.agents = {
            "status_agent": StatusAgent(self.amis_db),
            "doc_agent": DocumentAgent(self.amis_db, self.policy_kb),
            "booking_agent": BookingAgent(self.amis_db),
            "policy_agent": PolicyAgent(self.policy_kb)
        }
        
        # System metrics
        self.total_queries = 0
        self.successful_automations = 0
        self.escalations = 0
        self.start_time = time.time()
        
        logger.info("ResiBot 2.0 system initialized successfully")
    
    async def process_query(self, message: str, case_id: str = "12345") -> ChatResponse:
        """Main query processing pipeline"""
        start_time = time.time()
        system_logs = []
        
        try:
            self.total_queries += 1
            system_logs.append(f"[QUERY] Processing: '{message[:50]}...'")
            
            # Step 1: Intent Classification
            intent_result = self.intent_classifier.classify(message)
            system_logs.append(f"[INTENT] Classified as: {intent_result.intent.value} (confidence: {intent_result.confidence:.3f})")
            
            # Step 2: Confidence Check
            if intent_result.confidence < 0.60:
                self.escalations += 1
                system_logs.append(f"[ESCALATION] Low confidence, routing to human agent")
                
                return ChatResponse(
                    response="I'm not entirely sure how to help with your specific question. Let me connect you with a human caseworker who can provide detailed assistance. Please hold while I transfer your inquiry.",
                    intent=intent_result.intent.value,
                    confidence=intent_result.confidence,
                    agent_used="human_escalation",
                    processing_time=time.time() - start_time,
                    requires_escalation=True,
                    system_logs=system_logs
                )
            
            # Step 3: Agent Selection and Processing
            agent = self.agents.get(intent_result.agent)
            if not agent:
                system_logs.append(f"[ERROR] Agent not found: {intent_result.agent}")
                raise HTTPException(status_code=500, detail="Internal agent routing error")
            
            system_logs.append(f"[AGENT] Routing to: {agent.name}")
            
            # Step 4: Agent Processing
            agent_response = await agent.process(message, case_id, {})
            system_logs.append(f"[RESPONSE] Generated in {agent_response.processing_time:.3f}s")
            
            self.successful_automations += 1
            
            return ChatResponse(
                response=agent_response.response,
                intent=intent_result.intent.value,
                confidence=intent_result.confidence,
                agent_used=agent.name,
                processing_time=time.time() - start_time,
                requires_escalation=False,
                system_logs=system_logs
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            system_logs.append(f"[ERROR] {str(e)}")
            
            return ChatResponse(
                response="I encountered an error processing your request. Please try again or contact support if the issue persists.",
                intent="error",
                confidence=0.0,
                agent_used="error_handler",
                processing_time=time.time() - start_time,
                requires_escalation=True,
                system_logs=system_logs
            )
    
    def get_system_metrics(self) -> Dict:
        """Get current system performance metrics"""
        uptime = time.time() - self.start_time
        automation_rate = (self.successful_automations / self.total_queries * 100) if self.total_queries > 0 else 0
        
        return {
            "total_queries": self.total_queries,
            "successful_automations": self.successful_automations,
            "escalations": self.escalations,
            "automation_rate": round(automation_rate, 1),
            "uptime_seconds": round(uptime, 1),
            "agent_call_counts": {name: agent.call_count for name, agent in self.agents.items()}
        }

# ============================================================================
# FastAPI Web Interface
# ============================================================================

app = FastAPI(title="ResiBot 2.0 API", description="Technical Prototype API", version="1.0.0")
resibot = ResiBot()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    return await resibot.process_query(request.message, request.case_id or "12345")

@app.get("/metrics")
async def metrics_endpoint():
    """System metrics endpoint"""
    return resibot.get_system_metrics()

@app.get("/health")
async def health_endpoint():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "system": "ResiBot 2.0 Technical Prototype",
        "description": "RAG + MCP Agent Architecture for 90% Communication Automation",
        "endpoints": {
            "/chat": "Main chat interface",
            "/metrics": "System performance metrics",
            "/health": "Health check",
            "/docs": "API documentation"
        },
        "metrics": resibot.get_system_metrics()
    }

# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """Main function for CLI usage"""
    print("🤖 ResiBot 2.0 - Technical Prototype")
    print("=" * 50)
    print("Demonstrating RAG + MCP Agent Architecture")
    print("Type 'quit' to exit, 'metrics' for system stats")
    print("=" * 50)
    
    bot = ResiBot()
    
    # Demo queries
    demo_queries = [
        "Was ist der Status meiner Aufenthaltserlaubnis?",
        "What documents do I still need to submit?",
        "Wann ist mein nächster Termin?",
        "What are the requirements for §18a AufenthG?",
        "I have a general question about my case"
    ]
    
    print("\n📋 Try these demo queries:")
    for i, query in enumerate(demo_queries, 1):
        print(f"{i}. {query}")
    print()
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'metrics':
                metrics = bot.get_system_metrics()
                print("\n📊 System Metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
                continue
            elif not user_input:
                continue
            
            # Process query
            start_time = time.time()
            result = asyncio.run(bot.process_query(user_input))
            
            print(f"\n🤖 ResiBot: {result.response}")
            print(f"\n📈 Metrics: Intent={result.intent}, Confidence={result.confidence:.3f}, "
                  f"Agent={result.agent_used}, Time={result.processing_time:.3f}s")
            
            if result.requires_escalation:
                print("⚠️  This query requires human escalation")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Run FastAPI server
        uvicorn.run(
            "resibot:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    else:
        # Run CLI interface
        main()
