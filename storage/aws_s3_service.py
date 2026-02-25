"""
Nexus Trading System - AWS S3 Storage Service
Enterprise file storage and document management
"""

import boto3
import logging
from typing import Dict, List, Any, Optional, BinaryIO
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import os
import uuid
from botocore.exceptions import NoCredentialsError, ClientError

logger = logging.getLogger(__name__)

class FileType(Enum):
    """File types for storage categorization"""
    USER_DOCUMENTS = "user_documents"
    TRADING_REPORTS = "trading_reports"
    BACKUP_DATA = "backup_data"
    SYSTEM_LOGS = "system_logs"
    ANALYTICS_EXPORTS = "analytics_exports"
    USER_AVATARS = "user_avatars"
    TRADE_DOCUMENTS = "trade_documents"
    COMPLIANCE_FILES = "compliance_files"

class StorageTier(Enum):
    """Storage tiers for cost optimization"""
    STANDARD = "STANDARD"
    REDUCED_REDUNDANCY = "REDUCED_REDUNDANCY"
    STANDARD_IA = "STANDARD_IA"
    ONEZONE_IA = "ONEZONE_IA"
    INTELLIGENT_TIERING = "INTELLIGENT_TIERING"
    GLACIER = "GLACIER"
    DEEP_ARCHIVE = "DEEP_ARCHIVE"

@dataclass
class S3File:
    """S3 file metadata"""
    file_id: str
    file_name: str
    file_type: FileType
    file_size: int
    content_type: str
    bucket_name: str
    object_key: str
    storage_tier: StorageTier
    uploaded_at: datetime
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None

class S3StorageService:
    """AWS S3 storage service for enterprise file management"""
    
    def __init__(self, aws_access_key_id: str, aws_secret_access_key: str, 
                 region_name: str = "us-east-1", bucket_name: str = None):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self.bucket_name = bucket_name or f"nexus-trading-storage-{uuid.uuid4().hex[:8]}"
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        
        # Initialize S3 resource
        self.s3_resource = boto3.resource(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        
        # File registry for tracking
        self.file_registry = {}
        
        logger.info("S3 storage service initialized")
    
    async def create_bucket(self) -> Dict[str, Any]:
        """Create S3 bucket"""
        try:
            if self.region_name == 'us-east-1':
                bucket = self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                bucket = self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region_name}
                )
            
            # Set bucket policy for public access if needed
            await self._set_bucket_policy()
            
            logger.info(f"Created S3 bucket: {self.bucket_name}")
            
            return {
                "success": True,
                "bucket_name": self.bucket_name,
                "region": self.region_name,
                "bucket": bucket
            }
            
        except Exception as e:
            logger.error(f"Error creating bucket: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _set_bucket_policy(self):
        """Set bucket policy for access control"""
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "PublicReadGetObject",
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": "s3:GetObject",
                    "Resource": f"arn:aws:s3:::{self.bucket_name}/*"
                }
            ]
        }
        
        try:
            self.s3_client.put_bucket_policy(
                Bucket=self.bucket_name,
                Policy=json.dumps(policy)
            )
        except Exception as e:
            logger.warning(f"Could not set bucket policy: {e}")
    
    async def upload_file(self, file_data: BinaryIO, file_name: str, file_type: FileType,
                         user_id: str = None, metadata: Dict[str, str] = None,
                         storage_tier: StorageTier = StorageTier.STANDARD) -> Dict[str, Any]:
        """Upload file to S3"""
        try:
            # Generate unique object key
            file_id = str(uuid.uuid4())
            object_key = f"{file_type.value}/{user_id or 'system'}/{file_id}_{file_name}"
            
            # Prepare metadata
            s3_metadata = {
                "file_id": file_id,
                "file_name": file_name,
                "file_type": file_type.value,
                "uploaded_at": datetime.utcnow().isoformat(),
                "user_id": user_id or "system"
            }
            
            if metadata:
                s3_metadata.update(metadata)
            
            # Determine content type
            content_type = self._get_content_type(file_name)
            
            # Upload file
            self.s3_client.upload_fileobj(
                file_data,
                self.bucket_name,
                object_key,
                ExtraArgs={
                    "ContentType": content_type,
                    "Metadata": s3_metadata,
                    "StorageClass": storage_tier.value
                }
            )
            
            # Get file size
            file_data.seek(0, 2)  # Seek to end
            file_size = file_data.tell()
            file_data.seek(0)  # Reset to beginning
            
            # Create file record
            s3_file = S3File(
                file_id=file_id,
                file_name=file_name,
                file_type=file_type,
                file_size=file_size,
                content_type=content_type,
                bucket_name=self.bucket_name,
                object_key=object_key,
                storage_tier=storage_tier,
                uploaded_at=datetime.utcnow(),
                user_id=user_id,
                metadata=metadata
            )
            
            # Add to registry
            self.file_registry[file_id] = s3_file
            
            logger.info(f"Uploaded file: {file_name} to {object_key}")
            
            return {
                "success": True,
                "file_id": file_id,
                "object_key": object_key,
                "file_url": f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{object_key}",
                "file_size": file_size,
                "uploaded_at": s3_file.uploaded_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def download_file(self, file_id: str) -> Dict[str, Any]:
        """Download file from S3"""
        try:
            s3_file = self.file_registry.get(file_id)
            if not s3_file:
                return {
                    "success": False,
                    "error": f"File not found: {file_id}"
                }
            
            # Download file
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_file.object_key
            )
            
            file_data = response['Body'].read()
            
            logger.info(f"Downloaded file: {s3_file.file_name}")
            
            return {
                "success": True,
                "file_data": file_data,
                "file_name": s3_file.file_name,
                "content_type": s3_file.content_type,
                "file_size": s3_file.file_size,
                "metadata": s3_file.metadata
            }
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def delete_file(self, file_id: str) -> Dict[str, Any]:
        """Delete file from S3"""
        try:
            s3_file = self.file_registry.get(file_id)
            if not s3_file:
                return {
                    "success": False,
                    "error": f"File not found: {file_id}"
                }
            
            # Delete file
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_file.object_key
            )
            
            # Remove from registry
            del self.file_registry[file_id]
            
            logger.info(f"Deleted file: {s3_file.file_name}")
            
            return {
                "success": True,
                "file_id": file_id,
                "deleted_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_files(self, file_type: FileType = None, user_id: str = None, 
                        prefix: str = None) -> Dict[str, Any]:
        """List files in S3"""
        try:
            # Build prefix
            if prefix:
                list_prefix = prefix
            elif file_type and user_id:
                list_prefix = f"{file_type.value}/{user_id}/"
            elif file_type:
                list_prefix = f"{file_type.value}/"
            else:
                list_prefix = ""
            
            # List objects
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=list_prefix
            )
            
            files = []
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Get object metadata
                    try:
                        obj_response = self.s3_client.head_object(
                            Bucket=self.bucket_name,
                            Key=obj['Key']
                        )
                        
                        file_info = {
                            "object_key": obj['Key'],
                            "file_size": obj['Size'],
                            "last_modified": obj['LastModified'].isoformat(),
                            "content_type": obj_response.get('ContentType', ''),
                            "metadata": obj_response.get('Metadata', {}),
                            "storage_class": obj_response.get('StorageClass', 'STANDARD')
                        }
                        
                        files.append(file_info)
                        
                    except Exception as e:
                        logger.warning(f"Could not get metadata for {obj['Key']}: {e}")
            
            logger.info(f"Listed {len(files)} files with prefix: {list_prefix}")
            
            return {
                "success": True,
                "files": files,
                "count": len(files),
                "prefix": list_prefix
            }
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_file_url(self, file_id: str, expiration: int = 3600) -> Dict[str, Any]:
        """Generate presigned URL for file access"""
        try:
            s3_file = self.file_registry.get(file_id)
            if not s3_file:
                return {
                    "success": False,
                    "error": f"File not found: {file_id}"
                }
            
            # Generate presigned URL
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_file.object_key
                },
                ExpiresIn=expiration
            )
            
            logger.info(f"Generated presigned URL for file: {s3_file.file_name}")
            
            return {
                "success": True,
                "file_url": url,
                "expires_in": expiration,
                "expires_at": (datetime.utcnow() + timedelta(seconds=expiration)).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating file URL: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def copy_file(self, file_id: str, destination_key: str) -> Dict[str, Any]:
        """Copy file to another location"""
        try:
            s3_file = self.file_registry.get(file_id)
            if not s3_file:
                return {
                    "success": False,
                    "error": f"File not found: {file_id}"
                }
            
            # Copy object
            copy_source = {
                'Bucket': self.bucket_name,
                'Key': s3_file.object_key
            }
            
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket_name,
                Key=destination_key
            )
            
            logger.info(f"Copied file from {s3_file.object_key} to {destination_key}")
            
            return {
                "success": True,
                "source_key": s3_file.object_key,
                "destination_key": destination_key,
                "copied_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error copying file: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def move_file(self, file_id: str, destination_key: str) -> Dict[str, Any]:
        """Move file to another location"""
        try:
            # Copy file
            copy_result = await self.copy_file(file_id, destination_key)
            
            if copy_result["success"]:
                # Delete original file
                delete_result = await self.delete_file(file_id)
                
                if delete_result["success"]:
                    logger.info(f"Moved file to {destination_key}")
                    return {
                        "success": True,
                        "destination_key": destination_key,
                        "moved_at": datetime.utcnow().isoformat()
                    }
                else:
                    return delete_result
            else:
                return copy_result
                
        except Exception as e:
            logger.error(f"Error moving file: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            # Get bucket size
            total_size = 0
            file_count = 0
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=self.bucket_name):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        total_size += obj['Size']
                        file_count += 1
            
            # Get bucket info
            try:
                bucket_info = self.s3_client.head_bucket(Bucket=self.bucket_name)
                bucket_region = self.s3_client.get_bucket_location(Bucket=self.bucket_name)['LocationConstraint'] or 'us-east-1'
            except:
                bucket_region = self.region_name
            
            stats = {
                "bucket_name": self.bucket_name,
                "region": bucket_region,
                "total_files": file_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "total_size_gb": round(total_size / (1024 * 1024 * 1024), 2),
                "file_types": {},
                "storage_tiers": {},
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Count by file type and storage tier
            for s3_file in self.file_registry.values():
                file_type = s3_file.file_type.value
                storage_tier = s3_file.storage_tier.value
                
                stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
                stats["storage_tiers"][storage_tier] = stats["storage_tiers"].get(storage_tier, 0) + 1
            
            logger.info(f"Retrieved storage stats for {self.bucket_name}")
            
            return {
                "success": True,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_content_type(self, file_name: str) -> str:
        """Get content type based on file extension"""
        extension = file_name.lower().split('.')[-1] if '.' in file_name else ''
        
        content_types = {
            'txt': 'text/plain',
            'html': 'text/html',
            'css': 'text/css',
            'js': 'application/javascript',
            'json': 'application/json',
            'xml': 'application/xml',
            'pdf': 'application/pdf',
            'doc': 'application/msword',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'xls': 'application/vnd.ms-excel',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'ppt': 'application/vnd.ms-powerpoint',
            'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'svg': 'image/svg+xml',
            'mp4': 'video/mp4',
            'mp3': 'audio/mpeg',
            'zip': 'application/zip',
            'csv': 'text/csv'
        }
        
        return content_types.get(extension, 'application/octet-stream')
    
    async def backup_to_glacier(self, file_id: str) -> Dict[str, Any]:
        """Move file to Glacier storage tier"""
        try:
            s3_file = self.file_registry.get(file_id)
            if not s3_file:
                return {
                    "success": False,
                    "error": f"File not found: {file_id}"
                }
            
            # Change storage class to Glacier
            self.s3_client.copy_object(
                Bucket=self.bucket_name,
                Key=s3_file.object_key,
                CopySource={
                    'Bucket': self.bucket_name,
                    'Key': s3_file.object_key
                },
                StorageClass='GLACIER',
                MetadataDirective='COPY'
            )
            
            # Update file record
            s3_file.storage_tier = StorageTier.GLACIER
            
            logger.info(f"Moved file {s3_file.file_name} to Glacier")
            
            return {
                "success": True,
                "file_id": file_id,
                "storage_tier": "GLACIER",
                "moved_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error moving file to Glacier: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Factory function
def create_s3_storage_service(aws_access_key_id: str, aws_secret_access_key: str,
                             region_name: str = "us-east-1", bucket_name: str = None) -> S3StorageService:
    """Create and return S3 storage service instance"""
    return S3StorageService(aws_access_key_id, aws_secret_access_key, region_name, bucket_name)
