import requests
import logging
import time
import traceback
import json
import os
import uuid
import asyncio
from google.cloud import storage
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from google.cloud import logging as cloud_logging

from agents.user_storygen_agent.user_storygen_agent import UserStoryAgentDev
from utils.utils import get_setup_details, process_config

PRODUCT_OWNER_API_BASE = "https://dev.appmod.ai/backend/product_owner"
PROJECTS_API_BASE = "https://dev.appmod.ai/backend"
DEFAULT_BUCKET = "creative-workspace"
DEFAULT_AGENT_ID = "69491988effeb7573f9961a7"
DEFAULT_CONCIERGE_ID = "60655536-0750-4c0f-9be1-68604e53af0f"

@dataclass
class ArtifactCreateRequest:
    """Data class for artifact creation request"""
    id: str
    artifactTitle: str
    artifactId: Optional[str] = None
    artifactTitleIDs: Optional[list] = None
    artifact_title_ids: Optional[list] = None
    artifactData: Optional[Dict] = None
    modeName: Optional[str] = None
    modeId: Optional[str] = None
    artifactSubTitle: Optional[str] = None
    isLoading: bool = False
    widgetName: Optional[str] = None
    userEmail: Optional[str] = None
    optionalData: Optional[Dict] = None
    projectId: Optional[str] = None
    project_id: Optional[str] = None
    displayName: Optional[str] = None
    created_at: Optional[str] = None 
    updated_at: Optional[str] = None 
    dataSources_files: Optional[Dict] = None
    
    def __post_init__(self):
        if hasattr(self, 'project_id') and self.project_id and not self.projectId:
            self.projectId = self.project_id


def build_user_query(metadata: Dict) -> str:
    """Construct an intelligent user query based on available context"""
    query_parts = []
    
    if metadata.get('feature_name'):
        query_parts.append(f"Generate user stories for the feature: {metadata['feature_name']}")
    else:
        query_parts.append("Generate user stories based on the provided context")
    
    if metadata.get('feature_description'):
        query_parts.append(f"\nFeature Description: {metadata['feature_description']}")
    
    if metadata.get('git_url'):
        query_parts.append(f"\nRepository: {metadata['git_url']}")
    elif metadata.get('repo_urls'):
        repos = metadata['repo_urls']
        if isinstance(repos, list):
            query_parts.append(f"\nRepositories: {', '.join(repos[:3])}")
            if len(repos) > 3:
                query_parts.append(f" and {len(repos) - 3} more")
    
    files = metadata.get('selected_file_urls', [])
    if files:
        file_types = []
        for f in files:
            if f.endswith('.pdf'):
                file_types.append('PDF document')
            elif f.endswith(('.doc', '.docx')):
                file_types.append('Word document')
            elif f.endswith(('.xls', '.xlsx')):
                file_types.append('Excel spreadsheet')
            elif f.endswith(('.png', '.jpg', '.jpeg')):
                file_types.append('image')
            elif f.endswith(('.mp3', '.wav', '.m4a')):
                file_types.append('audio recording')
        if file_types:
            unique_types = list(set(file_types))
            query_parts.append(f"\nAnalyze the provided {', '.join(unique_types)}")
    
    story_type = metadata.get('user_story_type', ['DOCUMENT_ENHANCEMENT'])
    if isinstance(story_type, list) and story_type:
        story_type = story_type[0]
    
    type_instructions = {
        'DOCUMENT_ENHANCEMENT': "Focus on extracting requirements, user flows, and system behaviors from the provided documents.",
        'NEW_FEATURE': "Focus on defining new feature requirements, user interactions, and acceptance criteria.",
        'BUG_FIX': "Focus on identifying the issue, expected behavior, and validation criteria.",
        'TECHNICAL_DEBT': "Focus on identifying technical improvements, refactoring needs, and architectural changes."
    }
    
    if story_type in type_instructions:
        query_parts.append(f"\n{type_instructions[story_type]}")
    
    if metadata.get('user_questions'):
        questions = metadata['user_questions']
        if isinstance(questions, list) and questions:
            query_parts.append("\nConsider these user questions:")
            for i, q in enumerate(questions[:3]):
                if isinstance(q, dict):
                    query_parts.append(f"  - {q.get('question', '')}")
                elif isinstance(q, str):
                    query_parts.append(f"  - {q}")
    
    return "\n".join(query_parts)


def format_artifact_to_target_structure(
    redis_key: str, 
    metadata: Dict,
    artifact_content: Dict,
    gcs_signed_url: Optional[str] = None,
    gcs_gs_url: Optional[str] = None
) -> Dict:
    """Format artifact data to match the target structure"""
    try:
        final_result = artifact_content.get('final_combined_result', {})
        
        user_story_snapshot_raw = final_result.get('mmvf_list', [])
        if isinstance(user_story_snapshot_raw, dict) and 'response' in user_story_snapshot_raw:
            user_story_snapshot_raw = user_story_snapshot_raw['response']
        elif not isinstance(user_story_snapshot_raw, list):
            user_story_snapshot_raw = [user_story_snapshot_raw] if user_story_snapshot_raw else []
        
        def remove_artifact_ids(obj):
            if isinstance(obj, dict):
                obj.pop('artifactId', None)
                for k, v in obj.items():
                    if k == 'sub_features' and isinstance(v, list):
                        for sub_feature in v:
                            remove_artifact_ids(sub_feature)
                    else:
                        remove_artifact_ids(v)
            elif isinstance(obj, list):
                for item in obj:
                    remove_artifact_ids(item)
            return obj
        
        user_story_snapshot = remove_artifact_ids(user_story_snapshot_raw.copy()) if user_story_snapshot_raw else []
        
        file_citations_raw = final_result.get('file_citations', [])
        appmod_citations_raw = final_result.get('appmod_citations', [])
        image_citations_raw = final_result.get('image_citations', [])
        audio_citations_raw = final_result.get('audio_citations', [])
        web_citations_raw = final_result.get('web_citations', [])
        
        formatted_image_citations = []
        if image_citations_raw:
            for i, citation in enumerate(image_citations_raw):
                if isinstance(citation, dict):
                    formatted_citation = json.loads(json.dumps(citation))
                    formatted_citation['citation_number'] = str(i + 1)
                    if 'customMetadata' not in formatted_citation:
                        formatted_citation['customMetadata'] = {}
                    formatted_citation['customMetadata']['type'] = 'image_citation'
                    if 'relevance_score' not in formatted_citation['customMetadata']:
                        formatted_citation['customMetadata']['relevance_score'] = 85.0
                    formatted_image_citations.append(formatted_citation)
        
        formatted_file_citations = []
        for i, citation in enumerate(file_citations_raw):
            if isinstance(citation, dict):
                formatted_citation = json.loads(json.dumps(citation))
                formatted_citation['citation_number'] = str(i + 1)
                if 'customMetadata' not in formatted_citation:
                    formatted_citation['customMetadata'] = {}
                formatted_citation['customMetadata']['type'] = 'book_citation_pdf'
                if 'highlighted_pdf_gsutil_url' in formatted_citation['customMetadata']:
                    formatted_citation['customMetadata']['has_highlighted_pdf'] = bool(formatted_citation['customMetadata']['highlighted_pdf_gsutil_url'])
                formatted_file_citations.append(formatted_citation)
        
        formatted_audio_citations = []
        for i, citation in enumerate(audio_citations_raw):
            if isinstance(citation, dict):
                formatted_citation = json.loads(json.dumps(citation))
                formatted_citation['citation_number'] = str(i + 1)
                formatted_audio_citations.append(formatted_citation)
        
        ai_reasonings_snapshot = []
        for i, reason in enumerate(final_result.get('ai_reasonings', [])):
            if isinstance(reason, dict):
                ai_reasonings_snapshot.append({
                    "id": i + 1,
                    "reason": reason.get('reason', ''),
                    "gap": reason.get('gap', ''),
                    "relevance_score": reason.get('relevance_score', 85.0)
                })
            elif isinstance(reason, str):
                ai_reasonings_snapshot.append({
                    "id": i + 1,
                    "reason": reason,
                    "gap": "",
                    "relevance_score": 85.0
                })
        
        selected_files = metadata.get('selected_file_urls', [])
        document_files, image_files, meeting_recordings = [], [], []
        
        for file_url in selected_files:
            file_lower = file_url.lower()
            clean_url = file_url.replace('gs://', '', 1) if file_url.startswith('gs://') else file_url
            
            if file_lower.endswith(('.mp3', '.wav', '.m4a', '.ogg', '.mp4')):
                meeting_recordings.append({
                    "gs_url": clean_url,
                    "transcript_gcs_url": f"{clean_url}.transcript.json" if not clean_url.endswith('.json') else clean_url
                })
            elif file_lower.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append(clean_url)
            elif file_lower.endswith(('.pdf', '.docx', '.txt', '.xlsx', '.xls')):
                document_files.append(clean_url)
        
        knowledge_base = []
        for cit in formatted_file_citations:
            custom_metadata = cit.get('customMetadata', {})
            knowledge_base.append(
                f"FILE PATH: {custom_metadata.get('file_name', 'Unknown file')}\n"
                f"CITATION_TYPE: DOCUMENT\nCONTENT:\n{custom_metadata.get('highlighted_text', '')}"
            )
        
        for cit in formatted_audio_citations:
            custom_metadata = cit.get('customMetadata', {})
            transcript_text = custom_metadata.get('highlighted_text', [])
            if transcript_text:
                if isinstance(transcript_text, list):
                    transcript_text = "\n".join(transcript_text)
                knowledge_base.append(
                    f"FILE PATH: {custom_metadata.get('file_name', 'Unknown file')}\n"
                    f"CITATION_TYPE: AUDIO_TRANSCRIPT\nCONTENT:\n{transcript_text}"
                )
        
        breadcrumbs = [
            {"id": "setup", "label": "Gather Context", "isActive": True, "isVisited": True},
            {"id": "addMoreDataSources", "label": "Add Data Sources", "isActive": True, "isVisited": True},
            {"id": "featurePlanningOverview", "label": "Overview", "isActive": True, "isVisited": True},
            {"id": "userStory", "label": "User Story", "isActive": True, "isVisited": False}
        ]
        
        timestamp = datetime.now().strftime("%y/%m/%d %H:%M:%S")
        artifact_uuid = str(uuid.uuid4())
        
        project_summary = final_result.get('project_analyzer_summary', '')
        if isinstance(project_summary, list):
            project_summary = json.dumps(project_summary)
        
        result = {
            "id": artifact_uuid,
            "artifactId": artifact_uuid,
            "artifactTitle": metadata.get("artifactTitle", "User Story Artifact"),
            "artifactTitleIDs": metadata.get("artifactTitleIDs", ["US-001"]),
            "projectId": metadata.get('project_id', metadata.get('projectId', '')),
            "artifactData": {
                "final_response_gs_url": gcs_gs_url or "",
                "citationsSnapshot": {
                    "appmod_citations": appmod_citations_raw, 
                    "image_citations": formatted_image_citations,
                    "audio_citations": formatted_audio_citations,
                    "file_citations": formatted_file_citations,
                    "file_image_data": None,
                    "rca_citations": [],
                    "web_citations": web_citations_raw,
                    "code_generated": []
                },
                "activePageSnapShot": "userStory",
                "aiReasoningsSnapshot": ai_reasonings_snapshot,
                "breadCrumbsSnapShot": breadcrumbs,
                "dataSources_files": {
                    "document": document_files,
                    "image": image_files,
                    "meetingRecording": meeting_recordings
                },
                "knowledgeBaseSnapshot": knowledge_base,
                "projectAnalyserSummarySnapshot": project_summary,
                "projectIdSnapshot": metadata.get('project_id', metadata.get('projectId', '')),
                "userStorySnapshot": user_story_snapshot,
                "gcs_urls": {
                    "signed_url": gcs_signed_url,
                    "gs_url": gcs_gs_url,
                    "pdf_url": final_result.get('gcs_gs_url_pdf_response', ''),
                    "docx_url": final_result.get('gcs_gs_url_docx_response', '')
                }
            },
            "artifactSubTitle": metadata.get("artifactSubTitle", "View User Stories"),
            "created_at": timestamp,
            "dataSources_files": {
                "document": document_files,
                "image": image_files,
                "meetingRecording": meeting_recordings
            },
            "isLoading": False,
            "modeId": metadata.get("modeId"),
            "modeName": metadata.get("mode_name", "Requirement AI"),
            "optionalData": metadata.get("optionalData", {}),
            "updated_at": timestamp,
            "userEmail": metadata.get("user_email", metadata.get("userEmailId", "")),
            "widgetName": metadata.get("widgetName", "user story")
        }
        
        return result
        
    except Exception as e:
        logging.error(f"[Request id:{redis_key}] Error formatting artifact: {str(e)}")
        raise


def extract_artifact_from_agent(agent, response_text, redis_key, metadata, trace=None):
    """Extract and structure artifact data from the user story agent"""
    try:
        gcs_signed_url = getattr(agent, 'gcs_signed_url_final_response', None)
        gcs_gs_url = getattr(agent, 'gcs_gs_url_final_response', None)
        gcs_urls = {
            "pdf_url": getattr(agent, 'gcs_gs_url_pdf_response', None),
            "docx_url": getattr(agent, 'gcs_gs_url_docx_response', None)
        }
        
        artifact_content = {}
        if gcs_signed_url:
            try:
                response = requests.get(gcs_signed_url, timeout=30)
                if response.status_code == 200:
                    artifact_content = response.json()
                    if 'final_combined_result' in artifact_content:
                        artifact_content['final_combined_result'] = artifact_content['final_combined_result']
            except Exception as e:
                logging.error(f"[Request id:{redis_key}] Error fetching artifact from GCS: {str(e)}")
        
        artifact_payload = format_artifact_to_target_structure(
            redis_key=redis_key,
            metadata=metadata,
            artifact_content=artifact_content,
            gcs_signed_url=gcs_signed_url,
            gcs_gs_url=gcs_gs_url
        )
        
        artifact_payload.setdefault('artifactData', {})['gcs_urls'] = {
            "signed_url": gcs_signed_url,
            "gs_url": gcs_gs_url,
            **gcs_urls
        }
        
        if error := getattr(agent, 'error', None):
            artifact_payload['artifactData']['error'] = error
        
        return artifact_payload
        
    except Exception as e:
        logging.error(f"[Request id:{redis_key}] Error extracting artifact: {str(e)}")
        raise


async def upload_to_gcs(artifact_data: Dict[str, Any], project_id: str, artifact_id: str) -> Dict[str, Any]:
    """Upload artifact JSON to GCS"""
    try:
        bucket_name = os.environ.get("USER_STORY_GCS_BUCKET_NAME", DEFAULT_BUCKET)
        
        filename = f"Artifact_User Story_{artifact_id}.artifact.json"
        dest_blob_name = f"projects/{project_id}/{filename}"
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(dest_blob_name)
        
        if await asyncio.to_thread(blob.exists):
            timestamp = int(time.time())
            filename = f"Artifact_User Story_{artifact_id}_{timestamp}.artifact.json"
            dest_blob_name = f"projects/{project_id}/{filename}"
            blob = bucket.blob(dest_blob_name)
        
        json_bytes = json.dumps(artifact_data, indent=2).encode('utf-8')
        await asyncio.to_thread(
            blob.upload_from_string,
            data=json_bytes,
            content_type='application/json'
        )
        
        file_url = f"gs://{bucket_name}/{dest_blob_name}"
        signed_url = blob.generate_signed_url(expiration=3600)
        
        return {
            "success": True,
            "file_url": file_url,
            "signed_url": signed_url,
            "filename": filename,
            "gcs_path": dest_blob_name,
            "bucket": bucket_name
        }
        
    except Exception as e:
        error_msg = f"Error uploading to GCS: {str(e)}"
        logging.error(f"{error_msg}\n{traceback.format_exc()}")
        return {"success": False, "error": error_msg}
    

def strip_gs_prefix(file_url: str) -> str:
    """Remove gs:// prefix from GCS URL for project storage"""
    if file_url and file_url.startswith('gs://'):
        return file_url.replace('gs://', '', 1)
    return file_url


def create_artifact_in_mongodb_and_gcs(artifact_data: Dict[str, Any], redis_key: Optional[str] = None) -> Dict[str, Any]:
    """Create artifact via product_owner API and update project"""
    try:
        if 'projectId' not in artifact_data:
            if 'project_id' in artifact_data:
                artifact_data['projectId'] = artifact_data['project_id']
            elif 'artifactData' in artifact_data and 'projectIdSnapshot' in artifact_data['artifactData']:
                artifact_data['projectId'] = artifact_data['artifactData']['projectIdSnapshot']
        
        artifact = ArtifactCreateRequest(**artifact_data)
        artifact_title_ids = artifact.artifact_title_ids or artifact.artifactTitleIDs or []
        
        gcs_result = None
        if artifact.projectId:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                gcs_result = loop.run_until_complete(
                    upload_to_gcs(
                        artifact_data=artifact_data,
                        project_id=artifact.projectId,
                        artifact_id=artifact.id
                    )
                )
            finally:
                loop.close()
        
        artifact_api_payload = {
            "id": artifact.id,
            "artifactTitle": artifact.artifactTitle,
            "artifactData": artifact.artifactData or {},
            "modeName": artifact.modeName,
            "modeId": artifact.modeId,
            "artifactSubTitle": artifact.artifactSubTitle,
            "isLoading": artifact.isLoading if artifact.isLoading is not None else False,
            "widgetName": artifact.widgetName or "default-widget",
            "userEmail": artifact.userEmail,
            "optionalData": artifact.optionalData or {},
            "artifactTitleIDs": artifact_title_ids
        }
        
        if gcs_result and gcs_result.get("success"):
            artifact_api_payload.setdefault("artifactData", {})["gcs_signed_url"] = gcs_result["signed_url"]
        
        if artifact.optionalData and artifact.optionalData.get("userId"):
            artifact_api_payload["userId"] = artifact.optionalData["userId"]
        elif artifact_data.get("userId"):
            artifact_api_payload["userId"] = artifact_data.get("userId")
        
        response = requests.post(
            f"{PRODUCT_OWNER_API_BASE}/artifact",
            json=artifact_api_payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 400:
            error_detail = response.json().get("detail", "")
            if "Artifact already exists" in error_detail:
                return {"success": False, "error": "Artifact already exists", "artifact_id": artifact.id}
            return {"success": False, "error": f"API error: {error_detail}"}
        elif response.status_code == 422:
            return {"success": False, "error": f"Validation error: {response.json()}"}
        
        response.raise_for_status()
        result = response.json()
        
        if artifact.projectId and gcs_result and gcs_result.get("success"):
            try:
                project_response = requests.get(f"{PROJECTS_API_BASE}/projects/{artifact.projectId}")
                if project_response.status_code == 200:
                    current_project = project_response.json()
                    
                    project_file_url = strip_gs_prefix(gcs_result["file_url"])
                    
                    file_entry = {
                        "url": project_file_url,
                        "name": gcs_result["filename"],
                        "timestamp": datetime.utcnow().isoformat() + "000",
                        "isAI": True,
                        "type": "artifact",
                        "displayName": artifact.displayName or artifact.artifactTitle,
                        "modeId": artifact.modeId or "",
                        "modeName": artifact.modeName or "UserStory Mode",
                        "artifactCaption": "User Story Artifact",
                        "description": artifact.artifactSubTitle or "User story artifact generated by AI"
                    }
                    
                    current_files = current_project.get("files", [])
                    
                    if not any(f.get("url") == project_file_url for f in current_files):
                        existing_file_urls = current_project.get("file_urls", [])
                        cleaned_file_urls = [strip_gs_prefix(url) for url in existing_file_urls]
                        
                        if project_file_url not in cleaned_file_urls:
                            existing_file_urls.append(project_file_url)
                        
                        current_files.append(file_entry)
                        
                        update_payload = {
                            "name": current_project.get("name", ""),
                            "description": current_project.get("description", ""),
                            "user": current_project.get("user", []),
                            "context": current_project.get("context", {}),
                            "file_urls": existing_file_urls,
                            "parent": current_project.get("parent", ""),
                            "children": current_project.get("children", []),
                            "files": current_files,
                            "conversations": current_project.get("conversations", [])
                        }
                        
                        put_response = requests.put(
                            f"{PROJECTS_API_BASE}/projects/{artifact.projectId}",
                            json=update_payload,
                            headers={"Content-Type": "application/json"}
                        )
                        put_response.raise_for_status()
            except Exception as e:
                logging.error(f"[Request id:{redis_key}] Error updating project: {str(e)}")
        
        response_data = {
            "success": True,
            "artifact_id": artifact.id,
            "artifact_title_ids": artifact_title_ids,
            "mongo_id": result.get("artifact_id") or result.get("id") or artifact.id
        }
        
        if gcs_result and gcs_result.get("success"):
            response_data.update({
                "gcs_file_url": gcs_result["file_url"],
                "gcs_signed_url": gcs_result["signed_url"],
                "gcs_upload_success": True,
                "gcs_bucket": gcs_result["bucket"],
                "gcs_path": gcs_result["gcs_path"]
            })
        else:
            response_data["gcs_upload_success"] = False
            if gcs_result and gcs_result.get("error"):
                response_data["gcs_error"] = gcs_result["error"]
        
        return response_data
        
    except requests.exceptions.RequestException as e:
        error_msg = f"API request error: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_msg = f"API error ({e.response.status_code}): {e.response.json()}"
            except:
                error_msg = f"API error ({e.response.status_code}): {e.response.text}"
        
        logging.error(f"[Request id:{redis_key}] {error_msg}")
        return {"success": False, "error": error_msg}
        
    except Exception as e:
        logging.error(f"[Request id:{redis_key}] Error creating artifact: {str(e)}")
        return {"success": False, "error": f"Error creating artifact: {str(e)}"}


def generate_artifact(data, trace):
    """Main function to generate and save user story artifact"""
    redis_key = data.get('request_id')
    logging.info(f"[Request id:{redis_key}] Starting generate_artifact")
    
    try:
        metadata = data.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        user_query = build_user_query(metadata)
        
        agent_id = DEFAULT_AGENT_ID
        concierge_id = DEFAULT_CONCIERGE_ID
        
        data['prompt'] = user_query
        data['question'] = user_query
        data['modelType'] = 'gemini-2.5-flash'
        data['user_id'] = metadata.get('user_id', metadata.get('userId', 'system'))
        data['request_id'] = redis_key
        data['conversation_id'] = str(uuid.uuid4())
        data['chat_history'] = []
        
        mapped_metadata = {
            "user_query": user_query,
            "project_id": metadata.get("project_id", metadata.get("projectId", "")),
            "selected_file_urls": metadata.get("selectedFileUrls", metadata.get("selected_file_urls", [])),
            "user_id": metadata.get("user_id", metadata.get("userId", "system")),
            "user_email": metadata.get("userEmailId", metadata.get("user_email", "")),
            "mode_name": metadata.get("mode_name", metadata.get("modeName", "Requirement AI")),
            "user_story_type": metadata.get("user_story_type", metadata.get("userStoryType", ["DOCUMENT_ENHANCEMENT"])),
            "feature_name": metadata.get("feature_name", metadata.get("featureName")),
            "feature_description": metadata.get("feature_description", metadata.get("featureDescription")),
            "cra_redis_key": metadata.get("cra_redis_key", metadata.get("craRedisKey")),
            "audio_files_uploaded": metadata.get("audio_files_uploaded", metadata.get("audioFilesUploaded", False)),
            "git_token": metadata.get("git_token", metadata.get("gitToken", "")),
            "git_url": metadata.get("git_url", metadata.get("gitUrl", "")),
            "repo_urls": metadata.get("repo_urls", metadata.get("repoUrls", [])),
            "user_questions": metadata.get("user_questions", metadata.get("userQuestions", [])),
            "task_id": metadata.get("task_id", metadata.get("taskId", "")),
            "isAssistant": metadata.get("isAssistant", metadata.get("is_assistant", True)),
            "email_request_id": metadata.get("email_request_id", metadata.get("emailRequestId", "")),
            "chat_session_url": metadata.get("chat_session_url", metadata.get("chatSessionUrl", "")),
            "user_guide": metadata.get("user_guide", metadata.get("userGuide", "")),
            "artifactTitle": metadata.get("artifactTitle", "User Story Artifact"),
            "artifactSubTitle": metadata.get("artifactSubTitle", metadata.get("artifactSubtitle", "AI Generated User Stories")),
            "displayName": metadata.get("displayName", metadata.get("display_name", f"User Story Artifact - {datetime.now().strftime('%Y-%m-%d %H:%M')}")),
            "organization_name_list": metadata.get("organizationName", metadata.get("organization_name_list", [])),
            "file_instructions": metadata.get("file_instructions", metadata.get("fileInstructions", [])),
            "web_instructions": metadata.get("web_instructions", metadata.get("webInstructions", []))
        }
        
        if isinstance(mapped_metadata["organization_name_list"], str):
            try:
                mapped_metadata["organization_name_list"] = json.loads(mapped_metadata["organization_name_list"])
            except json.JSONDecodeError:
                mapped_metadata["organization_name_list"] = []
        
        config = get_setup_details(concierge_id, agent_id)
        if config is None:
            raise Exception("Could not fetch setup config from MongoDB")
        
        config['bucket_name'] = DEFAULT_BUCKET
        agent_config = process_config(config=config, sub_level="tools")
        
        user_story_agent = UserStoryAgentDev()
        setup_data = {
            **data,
            'prompt': user_query,
            'question': user_query,
            'user_id': mapped_metadata["user_id"],
            'request_id': redis_key,
            'chat_history': []
        }
        user_story_agent.setup(config=agent_config, data=setup_data)
        
        response_text = user_story_agent.run(
            user_query=user_query,
            files=mapped_metadata["selected_file_urls"],
            project_id=mapped_metadata["project_id"],
            user_guide=mapped_metadata["user_guide"],
            task_id=mapped_metadata["task_id"],
            git_url=mapped_metadata["git_url"],
            git_url_list=mapped_metadata["repo_urls"],
            user_id=mapped_metadata["user_id"],
            isAssistant=mapped_metadata["isAssistant"],
            email_request_id=mapped_metadata["email_request_id"],
            chat_session_url=mapped_metadata["chat_session_url"],
            mode_name=mapped_metadata["mode_name"],
            repo_urls=mapped_metadata["repo_urls"],
            user_questions=mapped_metadata["user_questions"],
            user_story_type=mapped_metadata["user_story_type"],
            feature_name=mapped_metadata["feature_name"],
            user_email=mapped_metadata["user_email"],
            git_token=mapped_metadata["git_token"],
            cra_redis_key=mapped_metadata["cra_redis_key"],
            organization_name_list=mapped_metadata["organization_name_list"],
            audio_files_uploaded=mapped_metadata["audio_files_uploaded"],
            next_trace={},
            trace=trace
        )
        
        if hasattr(user_story_agent, 'error') and user_story_agent.error:
            return {"success": False, "error": user_story_agent.error, "response": response_text}
        
        artifact_payload = extract_artifact_from_agent(
            agent=user_story_agent,
            response_text=response_text,
            redis_key=redis_key,
            metadata=mapped_metadata,
            trace=trace
        )
        
        save_result = create_artifact_in_mongodb_and_gcs(
            artifact_data=artifact_payload,
            redis_key=redis_key
        )
        
        return {
            "success": save_result.get("success", False),
            "response": response_text,
            "artifact_id": save_result.get("artifact_id"),
            "gcs_file_url": save_result.get("gcs_file_url"),
            "gcs_signed_url": save_result.get("gcs_signed_url"),
            "save_result": save_result
        }
        
    except Exception as e:
        logging.error(f"[Request id:{redis_key}] Error in generate_artifact: {str(e)}")
        return {"success": False, "error": str(e), "response": None}