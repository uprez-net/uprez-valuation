'use client';

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  File, 
  FileText, 
  Image, 
  X, 
  CheckCircle, 
  AlertCircle,
  Loader2,
  Download
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import * as Progress from '@radix-ui/react-progress';
import { cn, fileSize } from '@/lib/utils';
import { FileUploadProgress } from '@/types';

interface DocumentUploaderProps {
  onFilesUpload: (files: File[]) => void;
  acceptedTypes?: string[];
  maxFiles?: number;
  maxSize?: number; // in bytes
  projectId?: string;
  className?: string;
  multiple?: boolean;
  disabled?: boolean;
}

const fileTypeIcons = {
  'application/pdf': FileText,
  'application/vnd.ms-excel': FileText,
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': FileText,
  'application/msword': FileText,
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': FileText,
  'image/jpeg': Image,
  'image/png': Image,
  'image/gif': Image,
  'text/plain': File,
  'text/csv': FileText,
};

export function DocumentUploader({
  onFilesUpload,
  acceptedTypes = [
    'application/pdf',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/csv'
  ],
  maxFiles = 10,
  maxSize = 50 * 1024 * 1024, // 50MB
  projectId,
  className,
  multiple = true,
  disabled = false,
}: DocumentUploaderProps) {
  const [uploadProgress, setUploadProgress] = useState<FileUploadProgress[]>([]);
  const [isDragActive, setIsDragActive] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    // Handle rejected files
    if (rejectedFiles.length > 0) {
      const errors = rejectedFiles.map(({ file, errors }) => ({
        fileName: file.name,
        errors: errors.map((e: any) => e.message),
      }));
      
      console.warn('Rejected files:', errors);
      // You could show toast notifications here
    }

    // Process accepted files
    if (acceptedFiles.length > 0) {
      const newUploads: FileUploadProgress[] = acceptedFiles.map(file => ({
        file,
        progress: 0,
        status: 'pending',
      }));

      setUploadProgress(prev => [...prev, ...newUploads]);
      onFilesUpload(acceptedFiles);
    }
  }, [onFilesUpload]);

  const { getRootProps, getInputProps, isDragActive: dropzoneIsDragActive } = useDropzone({
    onDrop,
    accept: acceptedTypes.reduce((acc, type) => ({ ...acc, [type]: [] }), {}),
    maxFiles: multiple ? maxFiles : 1,
    maxSize,
    multiple,
    disabled,
    onDragEnter: () => setIsDragActive(true),
    onDragLeave: () => setIsDragActive(false),
  });

  const removeFile = (index: number) => {
    setUploadProgress(prev => prev.filter((_, i) => i !== index));
  };

  const retryUpload = (index: number) => {
    setUploadProgress(prev =>
      prev.map((item, i) =>
        i === index ? { ...item, status: 'pending', error: undefined } : item
      )
    );
    
    // Re-trigger upload
    onFilesUpload([uploadProgress[index].file]);
  };

  const getFileIcon = (file: File) => {
    const IconComponent = fileTypeIcons[file.type as keyof typeof fileTypeIcons] || File;
    return <IconComponent className="h-8 w-8 text-muted-foreground" />;
  };

  const getStatusIcon = (status: FileUploadProgress['status']) => {
    switch (status) {
      case 'uploading':
        return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return null;
    }
  };

  const updateProgress = (fileName: string, progress: number) => {
    setUploadProgress(prev =>
      prev.map(item =>
        item.file.name === fileName
          ? { ...item, progress, status: progress === 100 ? 'completed' : 'uploading' }
          : item
      )
    );
  };

  const setError = (fileName: string, error: string) => {
    setUploadProgress(prev =>
      prev.map(item =>
        item.file.name === fileName ? { ...item, status: 'error', error } : item
      )
    );
  };

  return (
    <div className={cn('space-y-4', className)}>
      {/* Drop Zone */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <Card
          {...getRootProps()}
          className={cn(
            'border-2 border-dashed transition-all duration-200 cursor-pointer',
            'hover:border-primary hover:bg-primary/5',
            (isDragActive || dropzoneIsDragActive) && 'border-primary bg-primary/10',
            disabled && 'opacity-50 cursor-not-allowed'
          )}
        >
          <CardContent className="flex flex-col items-center justify-center py-12 px-6">
            <input {...getInputProps()} />
            
            <motion.div
              animate={{
                scale: (isDragActive || dropzoneIsDragActive) ? 1.1 : 1,
                rotate: (isDragActive || dropzoneIsDragActive) ? 5 : 0,
              }}
              transition={{ duration: 0.2 }}
              className="mb-4"
            >
              <Upload className={cn(
                'h-12 w-12 transition-colors',
                (isDragActive || dropzoneIsDragActive) 
                  ? 'text-primary' 
                  : 'text-muted-foreground'
              )} />
            </motion.div>

            <div className="text-center space-y-2">
              <p className="text-lg font-medium">
                {(isDragActive || dropzoneIsDragActive)
                  ? 'Drop files here'
                  : 'Drop files or click to upload'
                }
              </p>
              <p className="text-sm text-muted-foreground">
                Upload prospectus, financial statements, or other documents
              </p>
              <p className="text-xs text-muted-foreground">
                Supported formats: PDF, Excel, Word, CSV • Max {fileSize(maxSize)} per file
                {multiple && ` • Up to ${maxFiles} files`}
              </p>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* File List */}
      <AnimatePresence>
        {uploadProgress.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="space-y-3"
          >
            {uploadProgress.map((item, index) => (
              <motion.div
                key={`${item.file.name}-${index}`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.2, delay: index * 0.1 }}
              >
                <Card className="p-4">
                  <div className="flex items-center space-x-4">
                    <div className="flex-shrink-0">
                      {getFileIcon(item.file)}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-sm font-medium truncate">
                          {item.file.name}
                        </p>
                        <div className="flex items-center space-x-2">
                          {getStatusIcon(item.status)}
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6"
                            onClick={() => removeFile(index)}
                          >
                            <X className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                      
                      <div className="flex items-center justify-between text-xs text-muted-foreground mb-2">
                        <span>{fileSize(item.file.size)}</span>
                        <span>{item.status === 'uploading' ? `${item.progress}%` : item.status}</span>
                      </div>

                      {/* Progress Bar */}
                      {item.status === 'uploading' && (
                        <Progress.Root
                          className="relative overflow-hidden bg-secondary rounded-full w-full h-2"
                          value={item.progress}
                        >
                          <Progress.Indicator
                            className="bg-primary w-full h-full transition-transform duration-300 ease-in-out"
                            style={{ transform: `translateX(-${100 - item.progress}%)` }}
                          />
                        </Progress.Root>
                      )}

                      {/* Error Message */}
                      {item.status === 'error' && item.error && (
                        <div className="flex items-center justify-between mt-2">
                          <p className="text-xs text-red-600">{item.error}</p>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => retryUpload(index)}
                            className="h-6 text-xs"
                          >
                            Retry
                          </Button>
                        </div>
                      )}

                      {/* Success Actions */}
                      {item.status === 'completed' && (
                        <div className="flex items-center justify-between mt-2">
                          <p className="text-xs text-green-600">Upload completed</p>
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-6 text-xs"
                          >
                            <Download className="h-3 w-3 mr-1" />
                            Download
                          </Button>
                        </div>
                      )}
                    </div>
                  </div>
                </Card>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );

  // Expose methods for parent components to control uploads
  React.useImperativeHandle(React.createRef(), () => ({
    updateProgress,
    setError,
  }));
}